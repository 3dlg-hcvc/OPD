import logging
import torch
from fvcore.nn import giou_loss, smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec, batched_nms, cat, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from typing import List, Optional, Union

import numpy as np
import math

# Use schduled sampling for training residual
def inverse_sigmoid_decay(i):
    # Specified for total epochs 30000
    k = 1200
    return k / (k + np.exp(i / k))


def getFocalLength(FOV, height, width=None):
    # FOV is in radius, should be vertical angle
    if width == None:
        f = height / (2 * math.tan(FOV / 2))
        return f
    else:
        fx = height / (2 * math.tan(FOV / 2))
        fy = fx / height * width
        return (fx, fy)


# MotionNet Version: based on the FastRCNNOutputs
class MotionOutputs:
    """
    A class that stores information about outputs of a Fast R-CNN head.
    It provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    # Add mtype, morigin, maxis in to the parameter list
    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        # 9DOF Pose prediction
        pred_pdimension_residual,
        pred_ptrans_residual,
        pred_protx_cat,
        pred_protx_residual,
        pred_proty_cat,
        pred_proty_residual,
        pred_protz_cat,
        pred_protz_residual,
        pred_pose_rt,
        # Motion Prediction
        pred_mtype,
        pred_morigin_residual,
        pred_maxis_cat,
        pred_maxis_residual,
        # Part Pose hyperparameters
        DIMENSION_MEAN,
        ROTATION_BIN_NUM,
        COVER_VALUE,
        proposals,
        smooth_l1_beta=0.0,
        box_reg_loss_type="smooth_l1",
        box_reg_loss_weight=1.0,
        POSE_ITER=0,
        MOTION_ITER=0,
        CTPM =False,
        freeze_DET=False,
        motionnet_type=None,
    ):
        """
        Args:
            box2box_transform (Box2BosxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
                The total number of all instances must be equal to R.
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
            box_reg_loss_weight (float): Weight for box regression loss
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        # 9DOF Pose prediction
        self.pred_pdimension_residual = pred_pdimension_residual
        self.pred_ptrans_residual = pred_ptrans_residual
        self.pred_protx_cat = pred_protx_cat
        self.pred_protx_residual = pred_protx_residual
        self.pred_proty_cat = pred_proty_cat
        self.pred_proty_residual = pred_proty_residual
        self.pred_protz_cat = pred_protz_cat
        self.pred_protz_residual = pred_protz_residual
        self.pred_pose_rt = pred_pose_rt
        # Motion Prediction
        self.pred_mtype = pred_mtype
        self.pred_morigin_residual = pred_morigin_residual
        self.pred_maxis_cat = pred_maxis_cat
        self.pred_maxis_residual = pred_maxis_residual

        self.smooth_l1_beta = smooth_l1_beta
        self.box_reg_loss_type = box_reg_loss_type
        self.box_reg_loss_weight = box_reg_loss_weight
        # Part Pose hyperparameters
        self.DIMENSION_MEAN = DIMENSION_MEAN
        self.ROTATION_BIN_NUM = ROTATION_BIN_NUM
        self.COVER_VALUE = COVER_VALUE
        self.motionnet_type = motionnet_type
        bin_range = (
            2 * np.pi + (ROTATION_BIN_NUM - 1) * COVER_VALUE
        ) / ROTATION_BIN_NUM
        rotation_bin = []
        current_value = -np.pi
        for i in range(ROTATION_BIN_NUM):
            rotation_bin.append([current_value, current_value + bin_range])
            current_value = current_value + bin_range - COVER_VALUE
        rotation_bin[ROTATION_BIN_NUM - 1][1] = np.pi
        self.rotation_bin = torch.tensor(
            rotation_bin, device=self.pred_class_logits.device
        )

        self.image_shapes = [x.image_size for x in proposals]

        if len(proposals):
            box_type = type(proposals[0].proposal_boxes)
            # cat(..., dim=0) concatenates over all images in the batch
            self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
            assert (
                not self.proposals.tensor.requires_grad
            ), "Proposals should not require gradients!"

            # The following fields should exist only when training.
            if proposals[0].has("gt_boxes"):
                self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
                assert proposals[0].has("gt_classes")
                self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)
                # 9DOF Part Pose
                self.gt_dimensions_residual = cat(
                    [p.gt_dimensions_residual for p in proposals], dim=0
                )
                self.gt_translations = cat(
                    [p.gt_translations for p in proposals], dim=0
                )
                self.gt_orig_rot = cat([p.gt_orig_rot for p in proposals], dim=0)
                if self.motionnet_type == "PM":
                    self.gt_rxs_cat = cat([p.gt_rxs_cat for p in proposals], dim=0)
                    self.gt_rxs_residual = cat(
                        [p.gt_rxs_residual for p in proposals], dim=0
                    )
                    self.gt_rxs_cover = cat([p.gt_rxs_cover for p in proposals], dim=0)
                    self.gt_rys_cat = cat([p.gt_rys_cat for p in proposals], dim=0)
                    self.gt_rys_residual = cat(
                        [p.gt_rys_residual for p in proposals], dim=0
                    )
                    self.gt_rys_cover = cat([p.gt_rys_cover for p in proposals], dim=0)
                    self.gt_rzs_cat = cat([p.gt_rzs_cat for p in proposals], dim=0)
                    self.gt_rzs_residual = cat(
                        [p.gt_rzs_residual for p in proposals], dim=0
                    )
                    self.gt_rzs_cover = cat([p.gt_rzs_cover for p in proposals], dim=0)
                elif self.motionnet_type == "PM_V0":
                    self.gt_pose_rt = cat([p.gt_pose_rt for p in proposals], dim=0)
                    ## Calculate the camera intrinsic parameters (they are fixed in this project)
                    FOV = 50
                    img_width = 256
                    img_height = 256
                    self.fx, self.fy = getFocalLength(FOV / 180 * math.pi, img_height, img_width)
                    self.cy = img_height / 2
                    self.cx = img_width / 2
                # Motion Parameters
                self.gt_types = cat([p.gt_types for p in proposals], dim=0)
                self.gt_origins = cat([p.gt_origins for p in proposals], dim=0)
                self.gt_axises = cat([p.gt_axises for p in proposals], dim=0)
                self.gt_motion_valids = cat(
                    [p.gt_motion_valids for p in proposals], dim=0
                )
                # Use the background class update the motion valids, when the prediction is a background, then don't consider its loss
                box_dim = self.gt_boxes.tensor.size(1)  # 4 or 5
                cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
                bg_class_ind = self.pred_class_logits.shape[1] - 1
                # Get the background proposal index
                bg_inds = nonzero_tuple(self.gt_classes == bg_class_ind)[0]
                # The valid total number is for the images we care about the motion loss, the total number of background and foreground 
                self.valid_total_num = self.gt_classes[self.gt_motion_valids].shape[0]
                self.gt_motion_valids[bg_inds] = False

        else:
            self.proposals = Boxes(
                torch.zeros(0, 4, device=self.pred_proposal_deltas.device)
            )
        self._no_instances = len(proposals) == 0  # no instances found

        self.POSE_ITER = POSE_ITER
        self.MOTION_ITER = MOTION_ITER

        self.CTPM = CTPM
        self.freeze_DET = freeze_DET

    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        num_instances = self.gt_classes.numel()
        pred_classes = self.pred_class_logits.argmax(dim=1)
        bg_class_ind = self.pred_class_logits.shape[1] - 1

        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

        storage = get_event_storage()
        if num_instances > 0:
            storage.put_scalar("fast_rcnn/cls_accuracy", num_accurate / num_instances)
            if num_fg > 0:
                storage.put_scalar(
                    "fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg
                )
                storage.put_scalar(
                    "fast_rcnn/false_negative", num_false_negative / num_fg
                )

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * self.pred_class_logits.sum()
        else:
            self._log_accuracy()
            return F.cross_entropy(
                self.pred_class_logits, self.gt_classes, reduction="mean"
            )

    def box_reg_loss(self):
        """
        Compute the smooth L1 loss for box regression.

        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * self.pred_proposal_deltas.sum()

        box_dim = self.gt_boxes.tensor.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = nonzero_tuple(
            (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        )[0]
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(
                box_dim, device=device
            )

        if self.box_reg_loss_type == "smooth_l1":
            gt_proposal_deltas = self.box2box_transform.get_deltas(
                self.proposals.tensor, self.gt_boxes.tensor
            )
            loss_box_reg = smooth_l1_loss(
                self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
                gt_proposal_deltas[fg_inds],
                self.smooth_l1_beta,
                reduction="sum",
            )
        elif self.box_reg_loss_type == "giou":
            loss_box_reg = giou_loss(
                self._predict_boxes()[fg_inds[:, None], gt_class_cols],
                self.gt_boxes.tensor[fg_inds],
                reduction="sum",
            )
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")

        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        loss_box_reg = loss_box_reg * self.box_reg_loss_weight / self.gt_classes.numel()
        return loss_box_reg

    def _predict_boxes(self):
        """
        Returns:
            Tensor: A Tensors of predicted class-specific or class-agnostic boxes
                for all images in a batch. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        return self.box2box_transform.apply_deltas(
            self.pred_proposal_deltas, self.proposals.tensor
        )

    """
    A subclass is expected to have the following methods because
    they are used to query information about the head predictions.
    """

    # 9DOF Part Pose Loss
    def pdimension_loss(self):
        if self._no_instances:
            return 0.0 * self.pred_pdimension_residual.sum()
        storage = get_event_storage()
        current_iter = storage.iter
        # Judge the annotaion is valid (No other augmentation to the origin image)
        if (
            current_iter < self.POSE_ITER
            or self.pred_pdimension_residual[self.gt_motion_valids].size(0) == 0
        ):
            return 0.0 * self.pred_pdimension_residual.sum()
        else:
            return smooth_l1_loss(
                self.pred_pdimension_residual[self.gt_motion_valids],
                self.gt_dimensions_residual[self.gt_motion_valids],
                self.smooth_l1_beta,
                reduction="sum",
            ) / self.valid_total_num

    def ptrans_loss(self):
        if self._no_instances:
            return 0.0 * self.pred_ptrans_residual.sum()

        storage = get_event_storage()
        current_iter = storage.iter
        if (
            current_iter < self.POSE_ITER
            or self.pred_ptrans_residual[self.gt_motion_valids].size(0) == 0
        ):
            return 0.0 * self.pred_ptrans_residual.sum()
        else:
            if self.CTPM:
                decay = 0
            else:
                decay = inverse_sigmoid_decay(current_iter)
            """ Use ground truth residual more at the start """
            if np.random.random() < decay:
                self.pred_box = self.gt_boxes.tensor[self.gt_motion_valids]
                # Use residual between the translation and gt 2dbbx as gt
                x1 = self.gt_boxes.tensor[self.gt_motion_valids][:, 0]
                y1 = self.gt_boxes.tensor[self.gt_motion_valids][:, 1]
                x2 = self.gt_boxes.tensor[self.gt_motion_valids][:, 2]
                y2 = self.gt_boxes.tensor[self.gt_motion_valids][:, 3]
                gt_residual = torch.zeros(
                    self.pred_ptrans_residual[self.gt_motion_valids].size(),
                    device=self.pred_ptrans_residual.device,
                )
                # x and y is the difference between the center of the 2dbbx and translation x, y
                gt_residual[:, 0] = (
                    self.gt_translations[self.gt_motion_valids][:, 0] - (x1 + x2) / 2
                )
                gt_residual[:, 1] = (
                    self.gt_translations[self.gt_motion_valids][:, 1] - (y1 + y2) / 2
                )
                gt_residual[:, 2] = self.gt_translations[self.gt_motion_valids][:, 2]

                gt_residual[:, 0] /= 256
                gt_residual[:, 1] /= 256

                # smooth_l1_loss loss for the translation residual
                return smooth_l1_loss(
                    self.pred_ptrans_residual[self.gt_motion_valids],
                    gt_residual,
                    self.smooth_l1_beta,
                    reduction="sum",
                ) / self.valid_total_num
            else:
                # Get the bbx for gt class
                device = self.pred_ptrans_residual.device
                box_dim = self.gt_boxes.tensor.size(1)  # 4 or 5
                cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
                bg_class_ind = self.pred_class_logits.shape[1] - 1
                fg_inds = nonzero_tuple(
                    (self.gt_classes[self.gt_motion_valids] >= 0)
                    & (self.gt_classes[self.gt_motion_valids] < bg_class_ind)
                )[0]
                if cls_agnostic_bbox_reg:
                    # pred_proposal_deltas only corresponds to foreground class for agnostic
                    gt_class_cols = torch.arange(box_dim, device=device)
                else:
                    fg_gt_classes = self.gt_classes[self.gt_motion_valids][fg_inds]
                    gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(
                        box_dim, device=device
                    )
                pred_deltas = self.pred_proposal_deltas[self.gt_motion_valids][
                    fg_inds[:, None], gt_class_cols
                ].detach()
                # Use residual between the translation and pred 2dbbx as gt
                pred_box = self.box2box_transform.apply_deltas(
                    pred_deltas, self.proposals.tensor[self.gt_motion_valids][fg_inds]
                )
                self.pred_box = pred_box
                x1 = pred_box[:, 0]
                y1 = pred_box[:, 1]
                x2 = pred_box[:, 2]
                y2 = pred_box[:, 3]

                gt_residual = torch.zeros(
                    self.pred_ptrans_residual[self.gt_motion_valids].size(),
                    device=device,
                )
                # x and y is the difference between the center of the 2dbbx and translation x, y
                gt_residual[:, 0] = (
                    self.gt_translations[self.gt_motion_valids][:, 0] - (x1 + x2) / 2
                )
                gt_residual[:, 1] = (
                    self.gt_translations[self.gt_motion_valids][:, 1] - (y1 + y2) / 2
                )
                gt_residual[:, 2] = self.gt_translations[self.gt_motion_valids][:, 2]

                gt_residual[:, 0] /= 256
                gt_residual[:, 1] /= 256

                # smooth_l1_loss for the translation residual
                return smooth_l1_loss(
                    self.pred_ptrans_residual[self.gt_motion_valids],
                    gt_residual,
                    self.smooth_l1_beta,
                    reduction="sum",
                ) / self.valid_total_num

    def prot_loss(self):
        if self._no_instances:
            return (
                0.0 * self.pred_protx_cat.sum()
                + 0.0 * self.pred_protx_residual.sum()
                + 0.0 * self.pred_proty_cat.sum()
                + 0.0 * self.pred_proty_residual.sum()
                + 0.0 * self.pred_protz_cat.sum()
                + 0.0 * self.pred_protz_residual.sum()
            )
        storage = get_event_storage()
        current_iter = storage.iter
        # Judge the annotaion is valid (No other augmentation to the origin image)
        if (
            current_iter < self.POSE_ITER
            or self.pred_protx_cat[self.gt_motion_valids].size(0) == 0
        ):
            return (
                0.0 * self.pred_protx_cat.sum()
                + 0.0 * self.pred_protx_residual.sum()
                + 0.0 * self.pred_proty_cat.sum()
                + 0.0 * self.pred_proty_residual.sum()
                + 0.0 * self.pred_protz_cat.sum()
                + 0.0 * self.pred_protz_residual.sum()
            )
        else:
            # Add loss for rx (category & residual)
            loss = F.cross_entropy(
                self.pred_protx_cat[self.gt_motion_valids],
                self.gt_rxs_cat.long()[self.gt_motion_valids],
                reduction="sum",
            ) / self.valid_total_num
            ## Normalize; Multiply 2 is for cos and sin
            normalize_protx_residual = torch.zeros(
                self.pred_protx_residual[self.gt_motion_valids].size(),
                device=self.pred_protx_residual.device,
            )
            for i in range(self.ROTATION_BIN_NUM):
                temp = (
                    self.pred_protx_residual[self.gt_motion_valids][:, 2 * i] ** 2
                    + self.pred_protx_residual[self.gt_motion_valids][:, 2 * i + 1] ** 2
                ) ** 0.5
                normalize_protx_residual[:, 2 * i] = (
                    self.pred_protx_residual[self.gt_motion_valids][:, 2 * i] / temp
                )
                normalize_protx_residual[:, 2 * i + 1] = (
                    self.pred_protx_residual[self.gt_motion_valids][:, 2 * i + 1] / temp
                )
            loss += 2 * smooth_l1_loss(
                normalize_protx_residual[self.gt_rxs_cover[self.gt_motion_valids]],
                self.gt_rxs_residual[self.gt_motion_valids][
                    self.gt_rxs_cover[self.gt_motion_valids]
                ],
                self.smooth_l1_beta,
                reduction="sum",
            ) / self.valid_total_num

            # Add loss for ry
            loss += F.cross_entropy(
                self.pred_proty_cat[self.gt_motion_valids],
                self.gt_rys_cat.long()[self.gt_motion_valids],
                reduction="sum",
            ) / self.valid_total_num
            ## Normalize; Multiply 2 is for cos and sin
            normalize_proty_residual = torch.zeros(
                self.pred_proty_residual[self.gt_motion_valids].size(),
                device=self.pred_proty_residual.device,
            )
            for i in range(self.ROTATION_BIN_NUM):
                temp = (
                    self.pred_proty_residual[self.gt_motion_valids][:, 2 * i] ** 2
                    + self.pred_proty_residual[self.gt_motion_valids][:, 2 * i + 1] ** 2
                ) ** 0.5
                normalize_proty_residual[:, 2 * i] = (
                    self.pred_proty_residual[self.gt_motion_valids][:, 2 * i] / temp
                )
                normalize_proty_residual[:, 2 * i + 1] = (
                    self.pred_proty_residual[self.gt_motion_valids][:, 2 * i + 1] / temp
                )
            loss += 2 * smooth_l1_loss(
                normalize_proty_residual[self.gt_rys_cover[self.gt_motion_valids]],
                self.gt_rys_residual[self.gt_motion_valids][
                    self.gt_rys_cover[self.gt_motion_valids]
                ],
                self.smooth_l1_beta,
                reduction="sum",
            ) / self.valid_total_num

            # Add loss for rz
            loss += F.cross_entropy(
                self.pred_protz_cat[self.gt_motion_valids],
                self.gt_rzs_cat.long()[self.gt_motion_valids],
                reduction="sum",
            ) / self.valid_total_num
            ## Normalize; Multiply 2 is for cos and sin
            normalize_protz_residual = torch.zeros(
                self.pred_protz_residual[self.gt_motion_valids].size(),
                device=self.pred_protz_residual.device,
            )
            for i in range(self.ROTATION_BIN_NUM):
                temp = (
                    self.pred_protz_residual[self.gt_motion_valids][:, 2 * i] ** 2
                    + self.pred_protz_residual[self.gt_motion_valids][:, 2 * i + 1] ** 2
                ) ** 0.5
                normalize_protz_residual[:, 2 * i] = (
                    self.pred_protz_residual[self.gt_motion_valids][:, 2 * i] / temp
                )
                normalize_protz_residual[:, 2 * i + 1] = (
                    self.pred_protz_residual[self.gt_motion_valids][:, 2 * i + 1] / temp
                )
            loss += 2 * smooth_l1_loss(
                normalize_protz_residual[self.gt_rzs_cover[self.gt_motion_valids]],
                self.gt_rzs_residual[self.gt_motion_valids][
                    self.gt_rzs_cover[self.gt_motion_valids]
                ],
                self.smooth_l1_beta,
                reduction="sum",
            ) / self.valid_total_num

            # Average the loss for three angles
            return loss / 3

    def pose_rt_loss(self):
        if self._no_instances:
            return 0.0 * self.pred_pose_rt.sum()
        storage = get_event_storage()
        current_iter = storage.iter
        if (
            current_iter < self.POSE_ITER
            or self.pred_pose_rt[self.gt_motion_valids].size(0) == 0
        ):
            return 0.0 * self.pred_pose_rt.sum()
        else:
            return smooth_l1_loss(
                self.pred_pose_rt[self.gt_motion_valids],
                self.gt_pose_rt[self.gt_motion_valids],
                self.smooth_l1_beta,
                reduction="sum",
            )  / self.valid_total_num

    # Motion Loss
    def mtype_loss(self):
        if self._no_instances:
            return 0.0 * self.pred_mtype.sum()
        storage = get_event_storage()
        current_iter = storage.iter
        # If motion valids == False, then don't calculate the loss
        if (
            current_iter < self.MOTION_ITER
            or self.pred_mtype[self.gt_motion_valids].size(0) == 0
        ):
            return 0.0 * self.pred_mtype.sum()
        else:
            return F.cross_entropy(
                self.pred_mtype[self.gt_motion_valids],
                self.gt_types.long()[self.gt_motion_valids],
                reduction="sum",
            ) / self.valid_total_num

    def morigin_loss(self):
        if self._no_instances:
            return 0.0 * self.pred_morigin_residual.sum()
        storage = get_event_storage()
        current_iter = storage.iter
        # If motion valids == False, then don't calculate the loss
        if (
            current_iter < self.MOTION_ITER
            or self.pred_morigin_residual[self.gt_motion_valids].size(0) == 0
        ):
            return 0.0 * self.pred_morigin_residual.sum()
        else:
            if self.CTPM:
                decay = 0
            else:
                decay = inverse_sigmoid_decay(current_iter)

            """ Use ground truth residual more at the start """
            if np.random.random() < decay:
                # Difference between the gt 3dbbx translation (x, y in image coordinate, z in depth) and motion origin (similar x, y, z setting)
                # Only calculate loss for rotation motion
                rot_inds = (
                    (self.gt_types[self.gt_motion_valids] == 0).nonzero().unbind(1)[0]
                )
                if (
                    self.pred_morigin_residual[self.gt_motion_valids][rot_inds].size(0)
                    == 0
                ):
                    return 0.0 * self.pred_morigin_residual.sum()
                gt_residual = (
                    self.gt_origins[self.gt_motion_valids][rot_inds]
                    - self.gt_translations[self.gt_motion_valids][rot_inds]
                )

                gt_residual[:, 0] /= 256
                gt_residual[:, 1] /= 256

                #  smooth_l1_losss for the origin residual
                return smooth_l1_loss(
                    self.pred_morigin_residual[self.gt_motion_valids][rot_inds],
                    gt_residual,
                    self.smooth_l1_beta,
                    reduction="sum",
                ) / self.valid_total_num
            else:
                # Only calculate loss for rotation motion
                rot_inds = (
                    (self.gt_types[self.gt_motion_valids] == 0).nonzero().unbind(1)[0]
                )
                if (
                    self.pred_morigin_residual[self.gt_motion_valids][rot_inds].size(0)
                    == 0
                ):
                    return 0.0 * self.pred_morigin_residual.sum()
                if self.motionnet_type == "PM":
                    x1 = self.pred_box[rot_inds][:, 0]
                    y1 = self.pred_box[rot_inds][:, 1]
                    x2 = self.pred_box[rot_inds][:, 2]
                    y2 = self.pred_box[rot_inds][:, 3]

                    gt_pose_trans = torch.zeros(
                        self.pred_ptrans_residual[self.gt_motion_valids][rot_inds].size(),
                        device=self.pred_ptrans_residual.device,
                    )
                    gt_pose_trans[:, 0] = (
                        self.pred_ptrans_residual[self.gt_motion_valids][rot_inds][:, 0]
                        + (x1 + x2) / 2
                    ).detach()
                    gt_pose_trans[:, 1] = (
                        self.pred_ptrans_residual[self.gt_motion_valids][rot_inds][:, 1]
                        + (y1 + y2) / 2
                    ).detach()
                    gt_pose_trans[:, 2] = self.pred_ptrans_residual[self.gt_motion_valids][
                        rot_inds
                    ][:, 2].detach()

                    gt_residual = (
                        self.gt_origins[self.gt_motion_valids][rot_inds] - gt_pose_trans
                    )
                elif self.motionnet_type == "PM_V0":
                    pose_trans = self.pred_pose_rt[:, 9:12]
                    pose_trans[:, 0] = pose_trans[:, 0] * self.fx / (-pose_trans[:, 2]) + self.cx
                    pose_trans[:, 1] = -(pose_trans[:, 1] * self.fy / (-pose_trans[:, 2])) + self.cy
                    pose_trans[:, 2] = -pose_trans[:, 2]

                    gt_residual = (
                        self.gt_origins[self.gt_motion_valids][rot_inds] - pose_trans[self.gt_motion_valids][rot_inds]
                    )

                gt_residual[:, 0] /= 256
                gt_residual[:, 1] /= 256

                # smooth_l1_loss for the origin residual
                return smooth_l1_loss(
                    self.pred_morigin_residual[self.gt_motion_valids][rot_inds],
                    gt_residual,
                    self.smooth_l1_beta,
                    reduction="sum",
                ) / self.valid_total_num

    def maxis_loss(self):
        if self._no_instances:
            return (
                0.0 * self.pred_maxis_cat.sum() + 0.0 * self.pred_maxis_residual.sum()
            )
        storage = get_event_storage()
        current_iter = storage.iter
        # If motion valids == False, then don't calculate the loss
        if (
            current_iter < self.MOTION_ITER
            or self.pred_maxis_cat[self.gt_motion_valids].size(0) == 0
        ):
            return (
                0.0 * self.pred_maxis_cat.sum() + 0.0 * self.pred_maxis_residual.sum()
            )
        else:
            if self.CTPM:
                decay = 0
            else:
                decay = inverse_sigmoid_decay(current_iter)
            """ Use ground truth residual more at the start """
            if np.random.random() < decay:
                # Use the six directions of the bounding box as the category to classify the axis
                # This should be the same to apply the rotation matrix on 6 base vectors
                # 6 base: [0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]
                # theta size: [N, 3]
                theta = self.gt_orig_rot[self.gt_motion_valids]
                # theta is the euler angle [x, y, z], convert euler zyx into rotation matrix
                R_x = torch.eye(3).repeat(theta.size(0), 1, 1)
                R_x = R_x.to(theta.device)
                R_x[:, 1, 1] = torch.cos(theta[:, 0])
                R_x[:, 1, 2] = -torch.sin(theta[:, 0])
                R_x[:, 2, 1] = torch.sin(theta[:, 0])
                R_x[:, 2, 2] = torch.cos(theta[:, 0])
                R_y = torch.eye(3).repeat(theta.size(0), 1, 1)
                R_y = R_y.to(theta.device)
                R_y[:, 0, 0] = torch.cos(theta[:, 1])
                R_y[:, 0, 2] = torch.sin(theta[:, 1])
                R_y[:, 2, 0] = -torch.sin(theta[:, 1])
                R_y[:, 2, 2] = torch.cos(theta[:, 1])
                R_z = torch.eye(3).repeat(theta.size(0), 1, 1)
                R_z = R_z.to(theta.device)
                R_z[:, 0, 0] = torch.cos(theta[:, 2])
                R_z[:, 0, 1] = -torch.sin(theta[:, 2])
                R_z[:, 1, 0] = torch.sin(theta[:, 2])
                R_z[:, 1, 1] = torch.cos(theta[:, 2])
                rotation_matrix = torch.matmul(R_z, torch.matmul(R_y, R_x))
                # Get current base vectors
                raw_base_vectors = torch.tensor(
                    [[0, 0, 0, 0, 1, -1], [0, 0, 1, -1, 0, 0], [1, -1, 0, 0, 0, 0]],
                    dtype=torch.float32,
                    device=theta.device,
                )
                base_vectors = torch.matmul(rotation_matrix, raw_base_vectors)
                axis_distance = torch.zeros((theta.size(0), 6), device=theta.device)
                for i in range(6):
                    axis_distance[:, i] = (
                        (
                            (
                                base_vectors[:, :, i]
                                - self.gt_axises[self.gt_motion_valids]
                            )
                            ** 2
                        ).sum(1)
                    ) ** 0.5
                gt_maxis_cat = axis_distance.argmin(1)
                gt_maxis_residual = (
                    base_vectors[range(base_vectors.size(0)), :, gt_maxis_cat]
                    - self.gt_axises[self.gt_motion_valids]
                )

                loss = F.cross_entropy(
                    self.pred_maxis_cat[self.gt_motion_valids],
                    gt_maxis_cat.long(),
                    reduction="sum",
                ) / self.valid_total_num
                loss += smooth_l1_loss(
                    self.pred_maxis_residual[self.gt_motion_valids][
                        torch.arange(
                            gt_maxis_residual.size(0), device=gt_maxis_residual.device
                        ).unsqueeze(1),
                        torch.arange(3, device=gt_maxis_residual.device)
                        + 3 * gt_maxis_cat.unsqueeze(1),
                    ],
                    gt_maxis_residual,
                    self.smooth_l1_beta,
                    reduction="sum",
                ) / self.valid_total_num
                return loss
            else:
                if self.motionnet_type == "PM":
                    theta = torch.zeros(
                        (self.gt_rxs_cat[self.gt_motion_valids].size(0), 3),
                        device=self.gt_rxs_cat.device,
                    )
                    # Deal with the gt_x
                    ## Get the correct residual angle (radian) from cos and sin
                    ## Use cos to calculate the angle, if sin < 0, negative the angle
                    pred_x_cat = self.pred_protx_cat[self.gt_motion_valids].argmax(1)
                    ## Normalize; Multiply 2 is for cos and sin
                    normalize_protx_residual = torch.zeros(
                        self.pred_protx_residual[self.gt_motion_valids].size(),
                        device=self.pred_protx_residual.device,
                    )
                    for i in range(self.ROTATION_BIN_NUM):
                        temp = (
                            self.pred_protx_residual[self.gt_motion_valids][:, 2 * i] ** 2
                            + self.pred_protx_residual[self.gt_motion_valids][:, 2 * i + 1]
                            ** 2
                        ) ** 0.5
                        normalize_protx_residual[:, 2 * i] = (
                            self.pred_protx_residual[self.gt_motion_valids][:, 2 * i] / temp
                        )
                        normalize_protx_residual[:, 2 * i + 1] = (
                            self.pred_protx_residual[self.gt_motion_valids][:, 2 * i + 1]
                            / temp
                        )
                    x_residual = torch.acos(
                        normalize_protx_residual[range(pred_x_cat.size(0)), pred_x_cat * 2]
                    ).detach()
                    ## Judge the sin, if sin < 0, negative the angle
                    x_neg_sin_inds = (
                        (
                            normalize_protx_residual[
                                range(pred_x_cat.size(0)), pred_x_cat * 2 + 1
                            ]
                            < 0
                        )
                        .nonzero()
                        .unbind(1)[0]
                    )
                    x_negative = torch.ones(
                        normalize_protx_residual.size(0), device=self.pred_protx_cat.device,
                    )
                    x_negative[x_neg_sin_inds] = -1
                    modified_x_residual = x_residual * x_negative
                    raw_x = (
                        modified_x_residual
                        + (
                            self.rotation_bin[pred_x_cat, 0]
                            + self.rotation_bin[pred_x_cat, 1]
                        )
                        / 2
                    )
                    proper_inds = (
                        ((raw_x >= -np.pi) & (raw_x <= np.pi)).nonzero().unbind(1)[0]
                    )
                    small_inds = (raw_x < -np.pi).nonzero().unbind(1)[0]
                    big_inds = (raw_x > np.pi).nonzero().unbind(1)[0]
                    theta[:, 0][proper_inds] = raw_x[proper_inds]
                    theta[:, 0][small_inds] = raw_x[small_inds] + 2 * np.pi
                    theta[:, 0][big_inds] = raw_x[big_inds] - 2 * np.pi
                    # Deal with the gt_y
                    pred_y_cat = self.pred_proty_cat[self.gt_motion_valids].argmax(1)
                    ## Normalize; Multiply 2 is for cos and sin
                    normalize_proty_residual = torch.zeros(
                        self.pred_proty_residual[self.gt_motion_valids].size(),
                        device=self.pred_proty_residual.device,
                    )
                    for i in range(self.ROTATION_BIN_NUM):
                        temp = (
                            self.pred_proty_residual[self.gt_motion_valids][:, 2 * i] ** 2
                            + self.pred_proty_residual[self.gt_motion_valids][:, 2 * i + 1]
                            ** 2
                        ) ** 0.5
                        normalize_proty_residual[:, 2 * i] = (
                            self.pred_proty_residual[self.gt_motion_valids][:, 2 * i] / temp
                        )
                        normalize_proty_residual[:, 2 * i + 1] = (
                            self.pred_proty_residual[self.gt_motion_valids][:, 2 * i + 1]
                            / temp
                        )
                    y_residual = torch.acos(
                        normalize_proty_residual[range(pred_y_cat.size(0)), pred_y_cat * 2]
                    ).detach()
                    y_neg_sin_inds = (
                        (
                            normalize_proty_residual[
                                range(pred_y_cat.size(0)), pred_y_cat * 2 + 1
                            ]
                            < 0
                        )
                        .nonzero()
                        .unbind(1)[0]
                    )
                    y_negative = torch.ones(
                        self.pred_proty_cat[self.gt_motion_valids].size(0),
                        device=self.pred_proty_cat.device,
                    )
                    y_negative[y_neg_sin_inds] = -1
                    modified_y_residual = y_residual * y_negative
                    raw_y = (
                        modified_y_residual
                        + (
                            self.rotation_bin[pred_y_cat, 0]
                            + self.rotation_bin[pred_y_cat, 1]
                        )
                        / 2
                    )
                    proper_inds = (
                        ((raw_y >= -np.pi) & (raw_y <= np.pi)).nonzero().unbind(1)[0]
                    )
                    small_inds = (raw_y < -np.pi).nonzero().unbind(1)[0]
                    big_inds = (raw_y > np.pi).nonzero().unbind(1)[0]
                    theta[:, 1][proper_inds] = raw_y[proper_inds]
                    theta[:, 1][small_inds] = raw_y[small_inds] + 2 * np.pi
                    theta[:, 1][big_inds] = raw_y[big_inds] - 2 * np.pi
                    # Deal with the gt_z
                    pred_z_cat = self.pred_protz_cat[self.gt_motion_valids].argmax(1)
                    ## Normalize; Multiply 2 is for cos and sin
                    normalize_protz_residual = torch.zeros(
                        self.pred_protz_residual[self.gt_motion_valids].size(),
                        device=self.pred_protz_residual.device,
                    )
                    for i in range(self.ROTATION_BIN_NUM):
                        temp = (
                            self.pred_protz_residual[self.gt_motion_valids][:, 2 * i] ** 2
                            + self.pred_protz_residual[self.gt_motion_valids][:, 2 * i + 1]
                            ** 2
                        ) ** 0.5
                        normalize_protz_residual[:, 2 * i] = (
                            self.pred_protz_residual[self.gt_motion_valids][:, 2 * i] / temp
                        )
                        normalize_protz_residual[:, 2 * i + 1] = (
                            self.pred_protz_residual[self.gt_motion_valids][:, 2 * i + 1]
                            / temp
                        )
                    z_residual = torch.acos(
                        normalize_protz_residual[range(pred_z_cat.size(0)), pred_z_cat * 2]
                    ).detach()
                    z_neg_sin_inds = (
                        (
                            normalize_protz_residual[
                                range(pred_z_cat.size(0)), pred_z_cat * 2 + 1
                            ]
                            < 0
                        )
                        .nonzero()
                        .unbind(1)[0]
                    )
                    z_negative = torch.ones(
                        self.pred_protz_cat[self.gt_motion_valids].size(0),
                        device=self.pred_protz_cat.device,
                    )
                    z_negative[z_neg_sin_inds] = -1
                    modified_z_residual = z_residual * z_negative
                    raw_z = (
                        modified_z_residual
                        + (
                            self.rotation_bin[pred_z_cat, 0]
                            + self.rotation_bin[pred_z_cat, 1]
                        )
                        / 2
                    )
                    proper_inds = (
                        ((raw_z >= -np.pi) & (raw_z <= np.pi)).nonzero().unbind(1)[0]
                    )
                    small_inds = (raw_z < -np.pi).nonzero().unbind(1)[0]
                    big_inds = (raw_z > np.pi).nonzero().unbind(1)[0]
                    theta[:, 2][proper_inds] = raw_z[proper_inds]
                    theta[:, 2][small_inds] = raw_z[small_inds] + 2 * np.pi
                    theta[:, 2][big_inds] = raw_z[big_inds] - 2 * np.pi
                    # theta is the euler angle [x, y, z], convert euler zyx into rotation matrix
                    R_x = torch.eye(3).repeat(theta.size(0), 1, 1)
                    R_x = R_x.to(theta.device)
                    R_x[:, 1, 1] = torch.cos(theta[:, 0])
                    R_x[:, 1, 2] = -torch.sin(theta[:, 0])
                    R_x[:, 2, 1] = torch.sin(theta[:, 0])
                    R_x[:, 2, 2] = torch.cos(theta[:, 0])
                    R_y = torch.eye(3).repeat(theta.size(0), 1, 1)
                    R_y = R_y.to(theta.device)
                    R_y[:, 0, 0] = torch.cos(theta[:, 1])
                    R_y[:, 0, 2] = torch.sin(theta[:, 1])
                    R_y[:, 2, 0] = -torch.sin(theta[:, 1])
                    R_y[:, 2, 2] = torch.cos(theta[:, 1])
                    R_z = torch.eye(3).repeat(theta.size(0), 1, 1)
                    R_z = R_z.to(theta.device)
                    R_z[:, 0, 0] = torch.cos(theta[:, 2])
                    R_z[:, 0, 1] = -torch.sin(theta[:, 2])
                    R_z[:, 1, 0] = torch.sin(theta[:, 2])
                    R_z[:, 1, 1] = torch.cos(theta[:, 2])
                    rotation_matrix = torch.matmul(R_z, torch.matmul(R_y, R_x))
                elif self.motionnet_type == "PM_V0":
                    rotation_matrix = self.pred_pose_rt[self.gt_motion_valids][:, :9].reshape((-1, 3, 3)).transpose(1, 2)
                # Get current base vectors
                raw_base_vectors = torch.tensor(
                    [[0, 0, 0, 0, 1, -1], [0, 0, 1, -1, 0, 0], [1, -1, 0, 0, 0, 0]],
                    dtype=torch.float32,
                    device=self.gt_orig_rot.device,
                )
                base_vectors = torch.matmul(rotation_matrix, raw_base_vectors)
                axis_distance = torch.zeros((self.gt_orig_rot[self.gt_motion_valids].size(0), 6), device=self.gt_orig_rot.device)
                for i in range(6):
                    axis_distance[:, i] = (
                        (
                            (
                                base_vectors[:, :, i]
                                - self.gt_axises[self.gt_motion_valids]
                            )
                            ** 2
                        ).sum(1)
                    ) ** 0.5
                gt_maxis_cat = axis_distance.argmin(1)
                gt_maxis_residual = (
                    base_vectors[range(base_vectors.size(0)), :, gt_maxis_cat]
                    - self.gt_axises[self.gt_motion_valids]
                )

                loss = F.cross_entropy(
                    self.pred_maxis_cat[self.gt_motion_valids],
                    gt_maxis_cat.long(),
                    reduction="sum",
                ) / self.valid_total_num
                loss += smooth_l1_loss(
                    self.pred_maxis_residual[self.gt_motion_valids][
                        torch.arange(
                            gt_maxis_residual.size(0), device=gt_maxis_residual.device
                        ).unsqueeze(1),
                        torch.arange(3, device=gt_maxis_residual.device)
                        + 3 * gt_maxis_cat.unsqueeze(1),
                    ],
                    gt_maxis_residual,
                    self.smooth_l1_beta,
                    reduction="sum",
                ) / self.valid_total_num
                return loss

    # MotionNet: log the motion accuracy
    @torch.no_grad()
    def _log_motion(self):
        num_instances = self.gt_types.numel()

        pred_probs = F.softmax(self.pred_mtype, dim=1)
        pred_types = torch.max(pred_probs, 1).indices.float()
        num_accurate_type = (pred_types == self.gt_types).nonzero().numel()

        storage = get_event_storage()
        if num_instances > 0:
            storage.put_scalar(
                "fast_rcnn/type_accuracy", num_accurate_type / num_instances
            )

    # MotionNet Version: add losses
    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        # MotionNet: log the motion accuracy for training dataset
        self._log_motion()

        if self.freeze_DET:
            return {
                "loss_cls": torch.tensor(0.0, device=self.gt_classes.device),
                "loss_box_reg": torch.tensor(0.0, device=self.gt_classes.device),
                # Pose Loss
                "loss_pdimension": self.pdimension_loss(),
                "loss_ptrans": self.ptrans_loss(),
                "loss_prot": self.prot_loss(),
                # Motion Loss
                "loss_mtype": self.mtype_loss(),
                "loss_morigin": self.morigin_loss(),
                "loss_maxis": self.maxis_loss(),
            }
        else:
            if self.motionnet_type == "PM":
                return {
                    "loss_cls": self.softmax_cross_entropy_loss(),
                    "loss_box_reg": self.box_reg_loss(),
                    # Pose Loss
                    "loss_pdimension": self.pdimension_loss(),
                    "loss_ptrans": self.ptrans_loss(),
                    "loss_prot": self.prot_loss(),
                    # Motion Loss
                    "loss_mtype": self.mtype_loss(),
                    "loss_morigin": self.morigin_loss(),
                    "loss_maxis": self.maxis_loss(),
                }
            elif self.motionnet_type == "PM_V0":
                return {
                    "loss_cls": self.softmax_cross_entropy_loss(),
                    "loss_box_reg": self.box_reg_loss(),
                    # Pose Loss
                    "loss_pdimension": self.pdimension_loss(),
                    "loss_pose_rt": self.pose_rt_loss(),
                    # Motion Loss
                    "loss_mtype": self.mtype_loss(),
                    "loss_morigin": self.morigin_loss(),
                    "loss_maxis": self.maxis_loss(),
                }


# MotionNet Version: baed on the MotionOutputLayers
class MotionOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    @configurable
    def __init__(
        self,
        input_shape,
        *,
        box2box_transform,
        num_classes,
        test_score_thresh=0.0,
        test_nms_thresh=0.5,
        test_topk_per_image=100,
        cls_agnostic_bbox_reg=False,
        smooth_l1_beta=0.0,
        box_reg_loss_type="smooth_l1",
        box_reg_loss_weight=1.0,
        DIMENSION_MEAN: Optional[list] = None,
        ROTATION_BIN_NUM: Optional[int] = None,
        COVER_VALUE: Optional[int] = None,
        use_GTBBX=False,
        use_GTPOSE=False,
        POSE_ITER=0,
        MOTION_ITER=0,
        use_GTCAT=False,
        CTPM=False,
        freeze_DET=False,
        motionnet_type=None,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss. Only used if
                `box_reg_loss_type` is "smooth_l1"
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
            box_reg_loss_weight (float): Weight for box regression loss
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = (
            input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        )

        # Part Pose hyperparameters
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.DIMENSION_MEAN = DIMENSION_MEAN
        self.ROTATION_BIN_NUM = ROTATION_BIN_NUM
        self.COVER_VALUE = COVER_VALUE
        self.motionnet_type = motionnet_type

        bin_range = (
            2 * np.pi + (ROTATION_BIN_NUM - 1) * COVER_VALUE
        ) / ROTATION_BIN_NUM
        rotation_bin = []
        current_value = -np.pi
        for i in range(ROTATION_BIN_NUM):
            rotation_bin.append([current_value, current_value + bin_range])
            current_value = current_value + bin_range - COVER_VALUE
        rotation_bin[ROTATION_BIN_NUM - 1][1] = np.pi
        self.rotation_bin = rotation_bin

        ## Calculate the camera intrinsic parameters (they are fixed in this project)
        FOV = 50
        img_width = 256
        img_height = 256
        self.fx, self.fy = getFocalLength(FOV / 180 * math.pi, img_height, img_width)
        self.cy = img_height / 2
        self.cx = img_width / 2

        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        """ Predict the part category """
        self.cls_score = nn.Sequential(
            Linear(input_size, 512),
            nn.ReLU(inplace=True),
            Linear(512, 128),
            nn.ReLU(inplace=True),
            Linear(128, 32),
            nn.ReLU(inplace=True),
            Linear(32, num_classes + 1),
        )
        for layer in self.cls_score:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
                )
                nn.init.constant_(layer.bias, 0)

        """ Predict the motion type """
        self.mtype_layer = nn.Sequential(
            Linear(input_size, 512),
            nn.LeakyReLU(inplace=True),
            Linear(512, 128),
            nn.LeakyReLU(inplace=True),
            Linear(128, 32),
            nn.LeakyReLU(inplace=True),
            Linear(32, 2),
        )
        for layer in self.mtype_layer:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
                )
                nn.init.constant_(layer.bias, 0)

        """ Predict the 3DOF dimension residual """
        self.pdimension_layer = nn.Sequential(
            Linear(input_size, 512),
            nn.LeakyReLU(inplace=True),
            Linear(512, 128),
            nn.LeakyReLU(inplace=True),
            Linear(128, 32),
            nn.LeakyReLU(inplace=True),
            Linear(32, 3),
        )
        for layer in self.pdimension_layer:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
                )
                nn.init.constant_(layer.bias, 0)

        """ Predict the translation based feature, box_delta, translation residual, motion origin residual """
        # Get the translation feature
        self.trans_feature_layer = nn.Sequential(
            Linear(input_size, 512), nn.LeakyReLU(inplace=True),
        )
        nn.init.kaiming_normal_(
            self.trans_feature_layer[0].weight,
            a=0,
            mode="fan_in",
            nonlinearity="leaky_relu",
        )
        nn.init.constant_(self.trans_feature_layer[0].bias, 0)
        # Predict the box delta
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        self.box_dim = box_dim
        self.bbox_pred = nn.Sequential(
            Linear(input_size, 512),
            nn.ReLU(inplace=True),
            Linear(512, 128),
            nn.ReLU(inplace=True),
            Linear(128, 32),
            nn.ReLU(inplace=True),
            Linear(32, num_bbox_reg_classes * box_dim),
        )
        for layer in self.bbox_pred:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
                )
                nn.init.constant_(layer.bias, 0)
        # Get the translation feature 2
        self.trans_feature2_layer = nn.Sequential(
            Linear(512, 256), nn.LeakyReLU(inplace=True),
        )
        nn.init.kaiming_normal_(
            self.trans_feature2_layer[0].weight,
            a=0,
            mode="fan_in",
            nonlinearity="leaky_relu",
        )
        nn.init.constant_(self.trans_feature2_layer[0].bias, 0)
        if self.motionnet_type == "PM":
            # Predict the translation residual
            self.ptrans_layer = nn.Sequential(
                Linear(256, 128),
                nn.LeakyReLU(inplace=True),
                Linear(128, 32),
                nn.LeakyReLU(inplace=True),
                Linear(32, 3),
            )
            for layer in self.ptrans_layer:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(
                        layer.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
                    )
                    nn.init.constant_(layer.bias, 0)
        # Predict the motion origin residual
        self.morigin_layer = nn.Sequential(
            Linear(256, 128),
            nn.LeakyReLU(inplace=True),
            Linear(128, 32),
            nn.LeakyReLU(inplace=True),
            Linear(32, 3),
        )
        for layer in self.morigin_layer:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
                )
                nn.init.constant_(layer.bias, 0)

        """ Predict the rotation based feature, pose rotation x (category & residual), pose rotation y (category & residual), pose rotation z (category & residual), motion axis residual """
        # Get the rotation feature
        self.rot_feature_layer = nn.Sequential(
            Linear(input_size, 512), nn.LeakyReLU(inplace=True),
        )
        nn.init.kaiming_normal_(
            self.rot_feature_layer[0].weight,
            a=0,
            mode="fan_in",
            nonlinearity="leaky_relu",
        )
        nn.init.constant_(self.rot_feature_layer[0].bias, 0)
        if self.motionnet_type == "PM":
            # Pose rotation x (category & residual)
            self.protx_cat_layer = nn.Sequential(
                Linear(512, 128),
                nn.LeakyReLU(inplace=True),
                Linear(128, 32),
                nn.LeakyReLU(inplace=True),
                Linear(32, self.ROTATION_BIN_NUM),
            )
            for layer in self.protx_cat_layer:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(
                        layer.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
                    )
                    nn.init.constant_(layer.bias, 0)
            self.protx_residual_layer = nn.Sequential(
                Linear(512, 128),
                nn.LeakyReLU(inplace=True),
                Linear(128, 32),
                nn.LeakyReLU(inplace=True),
                Linear(32, 2 * self.ROTATION_BIN_NUM),
            )
            for layer in self.protx_residual_layer:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(
                        layer.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
                    )
                    nn.init.constant_(layer.bias, 0)
            # Pose rotation y (category & residual)
            self.proty_cat_layer = nn.Sequential(
                Linear(512, 128),
                nn.LeakyReLU(inplace=True),
                Linear(128, 32),
                nn.LeakyReLU(inplace=True),
                Linear(32, self.ROTATION_BIN_NUM),
            )
            for layer in self.proty_cat_layer:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(
                        layer.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
                    )
                    nn.init.constant_(layer.bias, 0)
            self.proty_residual_layer = nn.Sequential(
                Linear(512, 128),
                nn.LeakyReLU(inplace=True),
                Linear(128, 32),
                nn.LeakyReLU(inplace=True),
                Linear(32, 2 * self.ROTATION_BIN_NUM),
            )
            for layer in self.proty_residual_layer:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(
                        layer.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
                    )
                    nn.init.constant_(layer.bias, 0)
            # Pose rotation z (category & residual)
            self.protz_cat_layer = nn.Sequential(
                Linear(512, 128),
                nn.LeakyReLU(inplace=True),
                Linear(128, 32),
                nn.LeakyReLU(inplace=True),
                Linear(32, self.ROTATION_BIN_NUM),
            )
            for layer in self.protz_cat_layer:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(
                        layer.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
                    )
                    nn.init.constant_(layer.bias, 0)
            self.protz_residual_layer = nn.Sequential(
                Linear(512, 128),
                nn.LeakyReLU(inplace=True),
                Linear(128, 32),
                nn.LeakyReLU(inplace=True),
                Linear(32, 2 * self.ROTATION_BIN_NUM),
            )
            for layer in self.protz_residual_layer:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(
                        layer.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
                    )
                    nn.init.constant_(layer.bias, 0)
        if self.motionnet_type == "PM_V0":
            self.pose_rt_layer  = nn.Sequential(
                Linear(input_size, 512),
                nn.ReLU(inplace=True),
                Linear(512, 128),
                nn.ReLU(inplace=True),
                Linear(128, 32),
                nn.ReLU(inplace=True),
                Linear(32, 12),
            )
        # maxis_residual
        self.maxis_cat_layer = nn.Sequential(
            Linear(512, 128),
            nn.LeakyReLU(inplace=True),
            Linear(128, 32),
            nn.LeakyReLU(inplace=True),
            Linear(32, 6),
        )
        for layer in self.maxis_cat_layer:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
                )
                nn.init.constant_(layer.bias, 0)
        self.maxis_residual_layer = nn.Sequential(
            Linear(512, 128),
            nn.LeakyReLU(inplace=True),
            Linear(128, 32),
            nn.LeakyReLU(inplace=True),
            Linear(32, 6 * 3),
        )
        for layer in self.maxis_residual_layer:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
                )
                nn.init.constant_(layer.bias, 0)

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        self.box_reg_loss_weight = box_reg_loss_weight

        self.use_GTBBX = use_GTBBX
        self.use_GTPOSE = use_GTPOSE

        self.POSE_ITER = POSE_ITER
        self.MOTION_ITER = MOTION_ITER

        self.use_GTCAT = use_GTCAT
        self.CTPM = CTPM
        self.freeze_DET = freeze_DET

    @classmethod
    def from_config(cls, cfg, input_shape):
        if "USE_GTBBX" in cfg.MODEL:
            use_GTBBX = cfg.MODEL.USE_GTBBX
        else:
            use_GTBBX = False

        if "USE_GTCAT" in cfg.MODEL:
            use_GTCAT = cfg.MODEL.USE_GTCAT
        else:
            use_GTCAT = False

        if "USE_GTPOSE" in cfg.MODEL:
            use_GTPOSE = cfg.MODEL.USE_GTPOSE
        else:
            use_GTPOSE = False

        if "CTPM" in cfg.MODEL:
            CTPM = cfg.MODEL.CTPM
        else:
            CTPM = False

        if "FREEZE_DET" in cfg.MODEL:
            freeze_DET = cfg.MODEL.FREEZE_DET
        else:
            freeze_DET = False

        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(
                weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS
            ),
            # fmt: off
            "num_classes"           : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg" : cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta"        : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh"     : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh"       : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image"   : cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type"     : cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "box_reg_loss_weight"   : cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT,
            # fmt: on
            # Part Pose
            "DIMENSION_MEAN": cfg.INPUT.DIMENSION_MEAN,
            "ROTATION_BIN_NUM": cfg.INPUT.ROTATION_BIN_NUM,
            "COVER_VALUE": cfg.INPUT.COVER_VALUE,
            "use_GTBBX": use_GTBBX,
            "use_GTPOSE": use_GTPOSE,
            "POSE_ITER": cfg.MODEL.POSE_ITER,
            "MOTION_ITER": cfg.MODEL.MOTION_ITER,
            "use_GTCAT": use_GTCAT,

            "CTPM": CTPM,
            "freeze_DET": freeze_DET,
            "motionnet_type": cfg.MODEL.MOTIONNET.TYPE,
        }

    def forward(self, x):
        """
        Returns:
            Tensor: shape (N,K+1), scores for each of the N box. Each row contains the scores for
                K object categories and 1 background class.
            Tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4), or (N,4)
                for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        if self.freeze_DET:
            # Currently this code only works for PM, not for PM_V0
            scores = self.cls_score(x)
            mtype = self.mtype_layer(x.detach())
            pdimension_residual = self.pdimension_layer(x.detach())
            # translation_based: bbox, translation residual, motion origin residual
            transfeature = self.trans_feature_layer(x.detach())
            proposal_deltas = self.bbox_pred(x)
            transfeature2 = self.trans_feature2_layer(transfeature)
            ptrans_residual = self.ptrans_layer(transfeature2)
            morigin_residual = self.morigin_layer(transfeature2)
            # rotation_based: rotation_x, rotation_y, rotation_z, axis_residual
            rotfeature = self.rot_feature_layer(x.detach())
            protx_cat = self.protx_cat_layer(rotfeature)
            protx_residual = self.protx_residual_layer(rotfeature)
            proty_cat = self.proty_cat_layer(rotfeature)
            proty_residual = self.proty_residual_layer(rotfeature)
            protz_cat = self.protz_cat_layer(rotfeature)
            protz_residual = self.protz_residual_layer(rotfeature)
            maxis_cat = self.maxis_cat_layer(rotfeature)
            maxis_residual = self.maxis_residual_layer(rotfeature)
        else:
            scores = self.cls_score(x)
            mtype = self.mtype_layer(x)
            pdimension_residual = self.pdimension_layer(x)
            # translation_based: bbox, translation residual, motion origin residual
            transfeature = self.trans_feature_layer(x)
            proposal_deltas = self.bbox_pred(x)
            transfeature2 = self.trans_feature2_layer(transfeature)
            if self.motionnet_type == "PM":
                ptrans_residual = self.ptrans_layer(transfeature2)
            morigin_residual = self.morigin_layer(transfeature2)
            # rotation_based: rotation_x, rotation_y, rotation_z, axis_residual
            rotfeature = self.rot_feature_layer(x)
            if self.motionnet_type == "PM":
                protx_cat = self.protx_cat_layer(rotfeature)
                protx_residual = self.protx_residual_layer(rotfeature)
                proty_cat = self.proty_cat_layer(rotfeature)
                proty_residual = self.proty_residual_layer(rotfeature)
                protz_cat = self.protz_cat_layer(rotfeature)
                protz_residual = self.protz_residual_layer(rotfeature)
            maxis_cat = self.maxis_cat_layer(rotfeature)
            maxis_residual = self.maxis_residual_layer(rotfeature)
            if self.motionnet_type == "PM_V0":
                pose_rt = self.pose_rt_layer(x)

        if self.motionnet_type == "PM":
            return (
                scores,
                proposal_deltas,
                # 9DOF Pose prediction
                pdimension_residual,
                ptrans_residual,
                protx_cat,
                protx_residual,
                proty_cat,
                proty_residual,
                protz_cat,
                protz_residual,
                # Motion Prediction
                mtype,
                morigin_residual,
                maxis_cat,
                maxis_residual,
            )
        elif self.motionnet_type == "PM_V0":
            return (
                scores,
                proposal_deltas,
                # 9DOF Pose prediction
                pdimension_residual,
                pose_rt,
                # Motion Prediction
                mtype,
                morigin_residual,
                maxis_cat,
                maxis_residual,
            )

    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features
                that were used to compute predictions.
        """
        if self.motionnet_type == "PM":
            (
                scores,
                proposal_deltas,
                # Part Pose Prediction
                pdimension_residual,
                ptrans_residual,
                protx_cat,
                protx_residual,
                proty_cat,
                proty_residual,
                protz_cat,
                protz_residual,
                # Motion Prediction
                mtype,
                morigin_residual,
                maxis_cat,
                maxis_residual,
            ) = predictions
            pose_rt = None
        else:
            (
                scores,
                proposal_deltas,
                # 9DOF Pose prediction
                pdimension_residual,
                pose_rt,
                # Motion Prediction
                mtype,
                morigin_residual,
                maxis_cat,
                maxis_residual,
            ) = predictions
            ptrans_residual = None
            protx_cat = None
            protx_residual = None
            proty_cat = None
            proty_residual = None
            protz_cat = None
            protz_residual = None

        return MotionOutputs(
            self.box2box_transform,
            scores,
            proposal_deltas,
            # Part Pose Prediction
            pdimension_residual,
            ptrans_residual,
            protx_cat,
            protx_residual,
            proty_cat,
            proty_residual,
            protz_cat,
            protz_residual,
            pose_rt,
            # Motion Prediction
            mtype,
            morigin_residual,
            maxis_cat,
            maxis_residual,
            # Part Pose hyperparameters
            self.DIMENSION_MEAN,
            self.ROTATION_BIN_NUM,
            self.COVER_VALUE,
            proposals,
            self.smooth_l1_beta,
            self.box_reg_loss_type,
            self.box_reg_loss_weight,
            self.POSE_ITER,
            self.MOTION_ITER,
            self.CTPM,
            self.freeze_DET,
            self.motionnet_type,
        ).losses()

    # MotionNet: functions used when testing
    def inference(self, predictions, proposals):
        """
        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        if scores[0].size(0) == 0:
            pdim = (torch.zeros((0, 3)),)
            ptrans = (torch.zeros((0, 9)),)
            prot = (torch.zeros((0, 3)),)
            mtype = (torch.zeros((0)),)
            morigin = (torch.zeros((0, 9)),)
            maxis = (torch.zeros((0, 3)),)
        else:
            pdim, ptrans, prot = self.predict_poses(predictions, proposals)
            mtype, morigin, maxis = self.predict_motions(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return motion_inference(
            boxes,
            scores,
            pdim,
            ptrans,
            prot,
            mtype,
            morigin,
            maxis,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

    def predict_boxes_for_gt_classes(self, predictions, proposals):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        if self.motionnet_type == "PM":
            (
                scores,
                proposal_deltas,
                # Part Pose Prediction
                pdimension_residual,
                ptrans_residual,
                protx_cat,
                protx_residual,
                proty_cat,
                proty_residual,
                protz_cat,
                protz_residual,
                # Motion Prediction
                mtype,
                morigin_residual,
                maxis_cat,
                maxis_residual,
            ) = predictions
        else:
            (
                scores,
                proposal_deltas,
                # 9DOF Pose prediction
                pdimension_residual,
                pose_rt,
                # Motion Prediction
                mtype,
                morigin_residual,
                maxis_cat,
                maxis_residual,
            ) = predictions

        # If using gtbbx, then make the delta to zero
        if self.use_GTBBX:
            proposal_deltas = torch.zeros(
                proposal_deltas.size(), device=proposal_deltas.device
            )

        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        N, B = proposal_boxes.shape
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)

        K = predict_boxes.shape[1] // B
        if K > 1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = gt_classes.clamp_(0, K - 1)

            predict_boxes = predict_boxes.view(N, K, B)[
                torch.arange(N, dtype=torch.long, device=predict_boxes.device),
                gt_classes,
            ]
        num_prop_per_image = [len(p) for p in proposals]
        return predict_boxes.split(num_prop_per_image)

    def predict_boxes(self, predictions, proposals):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        if self.motionnet_type == "PM":
            (
                scores,
                proposal_deltas,
                # Part Pose Prediction
                pdimension_residual,
                ptrans_residual,
                protx_cat,
                protx_residual,
                proty_cat,
                proty_residual,
                protz_cat,
                protz_residual,
                # Motion Prediction
                mtype,
                morigin_residual,
                maxis_cat,
                maxis_residual,
            ) = predictions
        else:
            (
                scores,
                proposal_deltas,
                # 9DOF Pose prediction
                pdimension_residual,
                pose_rt,
                # Motion Prediction
                mtype,
                morigin_residual,
                maxis_cat,
                maxis_residual,
            ) = predictions

        # If using gtbbx, then make the delta to zero
        if self.use_GTBBX:
            proposal_deltas = torch.zeros(
                proposal_deltas.size(), device=proposal_deltas.device
            )

        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = [p.proposal_boxes for p in proposals]
        # Concatenate all Boxes in proposal_boxes into one box ('cat' is a class method of the Boxes class)
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)
        self.pred_boxes = predict_boxes
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(self, predictions, proposals):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        if self.motionnet_type == "PM":
            (
                scores,
                proposal_deltas,
                # Part Pose Prediction
                pdimension_residual,
                ptrans_residual,
                protx_cat,
                protx_residual,
                proty_cat,
                proty_residual,
                protz_cat,
                protz_residual,
                # Motion Prediction
                mtype,
                morigin_residual,
                maxis_cat,
                maxis_residual,
            ) = predictions
        else:
            (
                scores,
                proposal_deltas,
                # 9DOF Pose prediction
                pdimension_residual,
                pose_rt,
                # Motion Prediction
                mtype,
                morigin_residual,
                maxis_cat,
                maxis_residual,
            ) = predictions
        num_inst_per_image = [len(p) for p in proposals]
        probs = F.softmax(scores, dim=-1)

        if self.use_GTCAT and not probs.size(0) == 0:
            probs = torch.zeros(probs.size(), device=probs.device)
            gt_classes = cat([p.gt_classes for p in proposals], dim=0)
            probs[range(probs.size(0)), gt_classes] = 1

        return probs.split(num_inst_per_image, dim=0)

    # MotionNet: results for the 9DOF bbx
    def predict_poses(self, predictions, proposals):
        if self.motionnet_type == "PM":
            (
                scores,
                proposal_deltas,
                # Part Pose Prediction
                pdimension_residual,
                ptrans_residual,
                protx_cat,
                protx_residual,
                proty_cat,
                proty_residual,
                protz_cat,
                protz_residual,
                # Motion Prediction
                mtype,
                morigin_residual,
                maxis_cat,
                maxis_residual,
            ) = predictions
        else:
            (
                scores,
                proposal_deltas,
                # 9DOF Pose prediction
                pdimension_residual,
                pose_rt,
                # Motion Prediction
                mtype,
                morigin_residual,
                maxis_cat,
                maxis_residual,
            ) = predictions
        num_inst_per_image = [len(p) for p in proposals]
        flag = False
        if self.use_GTPOSE:
            try: 
                # Apply gt pose dimension
                gt_dimensions_residual = cat(
                    [p.gt_dimensions_residual for p in proposals], dim=0
                )
                pdimension_residual = gt_dimensions_residual

                # Apply gt pose translation
                gt_classes = cat([p.gt_classes for p in proposals], dim=0)
                # Some proposals are ignored or have a background class. Their gt_classes
                # cannot be used as index.
                K = 3 # three category
                gt_classes = gt_classes.clamp_(0, K - 1)
                gt_class_cols = 4 * gt_classes[:, None] + torch.arange(
                        4, device=gt_classes.device
                    )
                gt_boxes = self.pred_boxes[torch.tensor(range(gt_class_cols.size(0)))[:, None], gt_class_cols]
                gt_translations = cat(
                    [p.gt_translations for p in proposals], dim=0
                )
                x1 = gt_boxes[:, 0]
                y1 = gt_boxes[:, 1]
                x2 = gt_boxes[:, 2]
                y2 = gt_boxes[:, 3]

                gt_residual = torch.zeros(
                    ptrans_residual.size(),
                    device=gt_boxes.device,
                )
                gt_residual[:, 0] = (
                    gt_translations[:, 0] - (x1 + x2) / 2
                ) / 256
                gt_residual[:, 1] = (
                    gt_translations[:, 1] - (y1 + y2) / 2
                ) / 256
                gt_residual[:, 2] = gt_translations[:, 2]
                ptrans_residual = gt_residual

                # Apply gt pose rotation
                gt_orig_rot = cat([p.gt_orig_rot for p in proposals], dim=0)
                self.pred_rotation = gt_orig_rot
                prot = gt_orig_rot.split(num_inst_per_image, dim=0)
                flag = True
            except:
                # Some images don't have valid gt
                pass


        # Use the pose dimension (predicted residual + dimiension mean)
        pdim = (
            pdimension_residual
            + torch.tensor(
                self.DIMENSION_MEAN,
                dtype=torch.float32,
                device=pdimension_residual.device,
            )
        ).split(num_inst_per_image, dim=0)

        if self.motionnet_type == "PM":
            ptrans_residual[:, 0] *= 256
            ptrans_residual[:, 1] *= 256
            # Use the 3d pose translation (residual -> 2D -> 3D)
            ## Calculate the translation for each class and corresponding bbx
            ## Becasue for each class, it has one corresponding box
            pred_boxes = self.pred_boxes
            ### Minus the background
            cat_num = scores.size(1) - 1
            device = scores.device
            pred_translation = torch.zeros(
                (pred_boxes.size(0), cat_num * 3), device=device
            )
            for i in range(cat_num):
                x1 = pred_boxes[:, i * 4 + 0]
                y1 = pred_boxes[:, i * 4 + 1]
                x2 = pred_boxes[:, i * 4 + 2]
                y2 = pred_boxes[:, i * 4 + 3]
                pred_translation[:, i * 3 + 0] = (
                    (ptrans_residual[:, 0] + (x1 + x2) / 2 - self.cx)
                    * (ptrans_residual[:, 2])
                    / self.fx
                )
                pred_translation[:, i * 3 + 1] = (
                    (ptrans_residual[:, 1] + (y1 + y2) / 2 - self.cy)
                    * (- ptrans_residual[:, 2])
                    / self.fy
                )
                pred_translation[:, i * 3 + 2] = -ptrans_residual[:, 2]
            self.pred_translation = pred_translation


            ptrans = pred_translation.split(num_inst_per_image, dim=0)

            if flag == False:
                # Use the Euler angle ZYX
                ## Normalize; Multiply 2 is for cos and sin
                device = protx_residual.device
                theta = torch.zeros((protx_residual.size(0), 3), device=device,)
                rotation_bin = torch.tensor(self.rotation_bin, device=device)
                ### Deal with x

                pred_x_cat = protx_cat.argmax(1)

                normalize_protx_residual = torch.zeros(
                    protx_residual.size(), device=device,
                )
                for i in range(self.ROTATION_BIN_NUM):
                    temp = (
                        protx_residual[:, 2 * i] ** 2 + protx_residual[:, 2 * i + 1] ** 2
                    ) ** 0.5
                    normalize_protx_residual[:, 2 * i] = protx_residual[:, 2 * i] / temp
                    normalize_protx_residual[:, 2 * i + 1] = (
                        protx_residual[:, 2 * i + 1] / temp
                    )
                x_residual = torch.acos(
                    normalize_protx_residual[range(pred_x_cat.size(0)), pred_x_cat * 2]
                )
                ### Judge the sin, if sin < 0, negative the angle
                x_neg_sin_inds = (
                    (
                        normalize_protx_residual[
                            range(pred_x_cat.size(0)), pred_x_cat * 2 + 1
                        ]
                        < 0
                    )
                    .nonzero()
                    .unbind(1)[0]
                )
                x_negative = torch.ones(normalize_protx_residual.size(0), device=device,)
                x_negative[x_neg_sin_inds] = -1
                modified_x_residual = x_residual * x_negative
                raw_x = (
                    modified_x_residual
                    + (rotation_bin[pred_x_cat, 0] + rotation_bin[pred_x_cat, 1]) / 2
                )
                proper_inds = ((raw_x >= -np.pi) & (raw_x <= np.pi)).nonzero().unbind(1)[0]
                small_inds = (raw_x < -np.pi).nonzero().unbind(1)[0]
                big_inds = (raw_x > np.pi).nonzero().unbind(1)[0]
                theta[:, 0][proper_inds] = raw_x[proper_inds]
                theta[:, 0][small_inds] = raw_x[small_inds] + 2 * np.pi
                theta[:, 0][big_inds] = raw_x[big_inds] - 2 * np.pi
                ### Deal with y
                pred_y_cat = proty_cat.argmax(1)
                normalize_proty_residual = torch.zeros(
                    proty_residual.size(), device=device,
                )
                for i in range(self.ROTATION_BIN_NUM):
                    temp = (
                        proty_residual[:, 2 * i] ** 2 + proty_residual[:, 2 * i + 1] ** 2
                    ) ** 0.5
                    normalize_proty_residual[:, 2 * i] = proty_residual[:, 2 * i] / temp
                    normalize_proty_residual[:, 2 * i + 1] = (
                        proty_residual[:, 2 * i + 1] / temp
                    )
                y_residual = torch.acos(
                    normalize_proty_residual[range(pred_y_cat.size(0)), pred_y_cat * 2]
                )
                y_neg_sin_inds = (
                    (
                        normalize_proty_residual[
                            range(pred_y_cat.size(0)), pred_y_cat * 2 + 1
                        ]
                        < 0
                    )
                    .nonzero()
                    .unbind(1)[0]
                )
                y_negative = torch.ones(normalize_proty_residual.size(0), device=device,)
                y_negative[y_neg_sin_inds] = -1
                modified_y_residual = y_residual * y_negative
                raw_y = (
                    modified_y_residual
                    + (rotation_bin[pred_y_cat, 0] + rotation_bin[pred_y_cat, 1]) / 2
                )
                proper_inds = ((raw_y >= -np.pi) & (raw_y <= np.pi)).nonzero().unbind(1)[0]
                small_inds = (raw_y < -np.pi).nonzero().unbind(1)[0]
                big_inds = (raw_y > np.pi).nonzero().unbind(1)[0]
                theta[:, 1][proper_inds] = raw_y[proper_inds]
                theta[:, 1][small_inds] = raw_y[small_inds] + 2 * np.pi
                theta[:, 1][big_inds] = raw_y[big_inds] - 2 * np.pi
                # Deal with z
                pred_z_cat = protz_cat.argmax(1)
                ## Normalize; Multiply 2 is for cos and sin
                normalize_protz_residual = torch.zeros(
                    protz_residual.size(), device=device,
                )
                for i in range(self.ROTATION_BIN_NUM):
                    temp = (
                        protz_residual[:, 2 * i] ** 2 + protz_residual[:, 2 * i + 1] ** 2
                    ) ** 0.5
                    normalize_protz_residual[:, 2 * i] = protz_residual[:, 2 * i] / temp
                    normalize_protz_residual[:, 2 * i + 1] = (
                        protz_residual[:, 2 * i + 1] / temp
                    )
                z_residual = torch.acos(
                    normalize_protz_residual[range(pred_z_cat.size(0)), pred_z_cat * 2]
                )
                z_neg_sin_inds = (
                    (
                        normalize_protz_residual[
                            range(pred_z_cat.size(0)), pred_z_cat * 2 + 1
                        ]
                        < 0
                    )
                    .nonzero()
                    .unbind(1)[0]
                )
                z_negative = torch.ones(normalize_protz_residual.size(0), device=device,)
                z_negative[z_neg_sin_inds] = -1
                modified_z_residual = z_residual * z_negative
                raw_z = (
                    modified_z_residual
                    + (rotation_bin[pred_z_cat, 0] + rotation_bin[pred_z_cat, 1]) / 2
                )
                proper_inds = ((raw_z >= -np.pi) & (raw_z <= np.pi)).nonzero().unbind(1)[0]
                small_inds = (raw_z < -np.pi).nonzero().unbind(1)[0]
                big_inds = (raw_z > np.pi).nonzero().unbind(1)[0]
                theta[:, 2][proper_inds] = raw_z[proper_inds]
                theta[:, 2][small_inds] = raw_z[small_inds] + 2 * np.pi
                theta[:, 2][big_inds] = raw_z[big_inds] - 2 * np.pi
                self.pred_rotation = theta
                prot = theta.split(num_inst_per_image, dim=0)
        elif self.motionnet_type == "PM_V0":
            cat_num = scores.size(1) - 1
            self.pred_translation = pose_rt[:, 9:12].repeat(1, cat_num)
            ptrans = self.pred_translation.split(num_inst_per_image, dim=0)

            self.pred_rotation_matrix = pose_rt[:, :9].reshape((-1, 3, 3)).transpose(1, 2)
            pred_theta = batchRotationMatrixToEulerAngles(pose_rt[:, :9].reshape((-1, 3, 3)).transpose(1, 2))
            prot = pred_theta.split(num_inst_per_image, dim=0)
            

        return (pdim, ptrans, prot)

    # MotionNet: predict results for the motion information
    def predict_motions(self, predictions, proposals):
        if self.motionnet_type == "PM":
            (
                scores,
                proposal_deltas,
                # Part Pose Prediction
                pdimension_residual,
                ptrans_residual,
                protx_cat,
                protx_residual,
                proty_cat,
                proty_residual,
                protz_cat,
                protz_residual,
                # Motion Prediction
                mtype,
                morigin_residual,
                maxis_cat,
                maxis_residual,
            ) = predictions
        else:
            (
                scores,
                proposal_deltas,
                # 9DOF Pose prediction
                pdimension_residual,
                pose_rt,
                # Motion Prediction
                mtype,
                morigin_residual,
                maxis_cat,
                maxis_residual,
            ) = predictions
        num_inst_per_image = [len(p) for p in proposals]

        # Motion Type
        mtype = (mtype.argmax(1)).split(num_inst_per_image, dim=0)

        # Motion Origin, use the predicted pose translation to calculate the 3D origin
        device = morigin_residual.device
        cat_num = scores.size(1) - 1
        pred_trans = torch.zeros(self.pred_translation.size(), device=device)
        pred_origin = torch.zeros(self.pred_translation.size(), device=device)
        
        morigin_residual[:, 0] *= 256
        morigin_residual[:, 1] *= 256

        for i in range(cat_num):
            pred_trans[:, 3 * i + 0] = (
                self.pred_translation[:, 3 * i + 0]
                * self.fx
                / (-self.pred_translation[:, 3 * i + 2])
                + self.cx
            )
            pred_trans[:, 3 * i + 1] = (
                self.pred_translation[:, 3 * i + 1]
                * self.fy
                / self.pred_translation[:, 3 * i + 2]
                + self.cy
            )
            pred_trans[:, 3 * i + 2] = -self.pred_translation[:, 3 * i + 2]

            pred_origin_2d = pred_trans[:, 3 * i : 3 * i + 3] + morigin_residual

            pred_origin[:, 3 * i + 0] = (
                (pred_origin_2d[:, 0] - self.cx) * pred_origin_2d[:, 2] / self.fx
            )
            pred_origin[:, 3 * i + 1] = (
                (pred_origin_2d[:, 1] - self.cy) * (-pred_origin_2d[:, 2]) / self.fy
            )
            pred_origin[:, 3 * i + 2] = -pred_origin_2d[:, 2]
        morigin = pred_origin.split(num_inst_per_image, dim=0)

        if self.motionnet_type == "PM":
            # Motion Axis, use the six bases for 3d bbx to infer the motion axis
            theta = self.pred_rotation
            # theta is the euler angle [x, y, z], convert euler zyx into rotation matrix
            R_x = torch.eye(3).repeat(theta.size(0), 1, 1)
            R_x = R_x.to(theta.device)
            R_x[:, 1, 1] = torch.cos(theta[:, 0])
            R_x[:, 1, 2] = -torch.sin(theta[:, 0])
            R_x[:, 2, 1] = torch.sin(theta[:, 0])
            R_x[:, 2, 2] = torch.cos(theta[:, 0])
            R_y = torch.eye(3).repeat(theta.size(0), 1, 1)
            R_y = R_y.to(theta.device)
            R_y[:, 0, 0] = torch.cos(theta[:, 1])
            R_y[:, 0, 2] = torch.sin(theta[:, 1])
            R_y[:, 2, 0] = -torch.sin(theta[:, 1])
            R_y[:, 2, 2] = torch.cos(theta[:, 1])
            R_z = torch.eye(3).repeat(theta.size(0), 1, 1)
            R_z = R_z.to(theta.device)
            R_z[:, 0, 0] = torch.cos(theta[:, 2])
            R_z[:, 0, 1] = -torch.sin(theta[:, 2])
            R_z[:, 1, 0] = torch.sin(theta[:, 2])
            R_z[:, 1, 1] = torch.cos(theta[:, 2])
            rotation_matrix = torch.matmul(R_z, torch.matmul(R_y, R_x))
        elif self.motionnet_type == "PM_V0":
            rotation_matrix = self.pred_rotation_matrix
        # Get current base vectors
        raw_base_vectors = torch.tensor(
            [[0, 0, 0, 0, 1, -1], [0, 0, 1, -1, 0, 0], [1, -1, 0, 0, 0, 0]],
            dtype=torch.float32,
            device=rotation_matrix.device,
        )
        base_vectors = torch.matmul(rotation_matrix, raw_base_vectors)

        gt_maxis_cat = maxis_cat.argmax(1)
        inds = torch.arange(3, device=device) + 3 * gt_maxis_cat.unsqueeze(1)
        # To make it consistent with the loss (the residual is base_vector - gt_axis)
        gt_maxis = (
            base_vectors[
                (torch.arange(base_vectors.size(0))).unsqueeze(1),
                :,
                gt_maxis_cat.unsqueeze(1),
            ].squeeze(1)
            - maxis_residual[(torch.arange(maxis_residual.size(0))).unsqueeze(1), inds]
        )
        temp_sum = (
            gt_maxis[:, 0] ** 2 + gt_maxis[:, 1] ** 2 + gt_maxis[:, 2] ** 2
        ) ** 0.5
        gt_maxis = gt_maxis / temp_sum.unsqueeze(1)
        maxis = gt_maxis.split(num_inst_per_image, dim=0)

        return (mtype, morigin, maxis)


# MotionNet: based on fast_rcnn_inference
def motion_inference(
    boxes,
    scores,
    pdim,
    ptrans,
    prot,
    mtype,
    morigin,
    maxis,
    image_shapes,
    score_thresh,
    nms_thresh,
    topk_per_image,
):
    """
    Call `motion_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        motion_inference_single_image(
            boxes_per_image,
            scores_per_image,
            pdim_per_image,
            ptrans_per_image,
            prot_per_image,
            mtype_per_image,
            morigin_per_image,
            maxis_per_image,
            image_shape,
            score_thresh,
            nms_thresh,
            topk_per_image,
        )
        for scores_per_image, boxes_per_image, pdim_per_image, ptrans_per_image, prot_per_image, mtype_per_image, morigin_per_image, maxis_per_image, image_shape in zip(
            scores, boxes, pdim, ptrans, prot, mtype, morigin, maxis, image_shapes
        )
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


# MotionNet: based on fast_rcnn_inference_single_image
def motion_inference_single_image(
    boxes,
    scores,
    pdim,
    ptrans,
    prot,
    mtype,
    morigin,
    maxis,
    image_shape,
    score_thresh,
    nms_thresh,
    topk_per_image,
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        # MotionNet
        pdim = pdim[valid_mask]
        ptrans = ptrans[valid_mask]
        prot = prot[valid_mask]
        mtype = mtype[valid_mask]
        morigin = morigin[valid_mask]
        maxis = maxis[valid_mask]

    # Don't care the score for the background when evaluation
    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4
    # MotionNet
    ptrans = ptrans.view(-1, num_bbox_reg_classes, 3)
    morigin = morigin.view(-1, num_bbox_reg_classes, 3)

    # Filter results based on detection scores
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
        # MotionNet
        ptrans = ptrans[filter_inds[:, 0], 0]
        morigin = morigin[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
        # MotionNet
        ptrans = ptrans[filter_mask]
        morigin = morigin[filter_mask]
    scores = scores[filter_mask]
    # MotionNet
    pdim = pdim[filter_inds[:, 0]]
    prot = prot[filter_inds[:, 0]]
    mtype = mtype[filter_inds[:, 0]]
    maxis = maxis[filter_inds[:, 0]]

    # Apply per-class NMS
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds, pdim, ptrans, prot, mtype, morigin, maxis = (
        boxes[keep],
        scores[keep],
        filter_inds[keep],
        pdim[keep],
        ptrans[keep],
        prot[keep],
        mtype[keep],
        morigin[keep],
        maxis[keep],
    )

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    # MotionNet
    result.pdim = pdim
    result.ptrans = ptrans
    result.prot = prot
    result.mtype = mtype
    result.morigin = morigin
    result.maxis = maxis
    # pred_classes only have foreground class
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]

# Calculates rotation matrix to euler angles
# The reuslt is for euler angles (ZYX) radians
def batchRotationMatrixToEulerAngles(R):
    sy = torch.sqrt(R[:, 0, 0] * R[:, 0, 0] + R[:, 1, 0] * R[:, 1, 0])

    singular = sy < 1e-6
    non_singular = torch.logical_not(singular)
    
    theta = torch.zeros((R.shape[0], 3), device=R.device)
    theta[non_singular, 0] = torch.atan2(R[non_singular][:, 2, 1], R[non_singular][:, 2, 2])
    theta[non_singular, 1] = torch.atan2(-R[non_singular][:, 2, 0], sy[non_singular])
    theta[non_singular, 2] = torch.atan2(R[non_singular][:, 1, 0], R[non_singular][:, 0, 0])

    theta[singular, 0] = torch.atan2(-R[singular][:, 1, 2], R[singular][:, 1, 1])
    theta[singular, 1] = torch.atan2(-R[singular][:, 2, 0], sy[singular])
    theta[singular, 2] = 0

    return theta