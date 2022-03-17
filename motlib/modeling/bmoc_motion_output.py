import logging
import torch
from fvcore.nn import giou_loss, smooth_l1_loss
from torch import nn
from torch.nn import functional as F
import numpy as np
import json

from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec, batched_nms, cat, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference


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
        pred_mtype,
        pred_morigin,
        pred_maxis,
        pred_extrinsic,
        pred_mstate,
        proposals,
        smooth_l1_beta=0.0,
        box_reg_loss_type="smooth_l1",
        box_reg_loss_weight=1.0,
        freeze_DET=False,
        motionnet_type=None,
        motionstate=False,
        state_bgfg=False,
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
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
        # Three prediction for motion
        self.pred_mtype = pred_mtype
        self.pred_morigin = pred_morigin
        self.pred_maxis = pred_maxis
        self.pred_extrinsic = pred_extrinsic
        self.pred_mstate = pred_mstate

        self.smooth_l1_beta = smooth_l1_beta
        self.box_reg_loss_type = box_reg_loss_type
        self.box_reg_loss_weight = box_reg_loss_weight

        self.motionnet_type = motionnet_type
        self.motionstate = motionstate

        self.state_bgfg = state_bgfg

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
                if self.motionstate:
                    self.gt_states = cat([p.gt_states for p in proposals], dim=0)
                self.gt_types = cat([p.gt_types for p in proposals], dim=0)
                self.gt_origins = cat([p.gt_origins for p in proposals], dim=0)
                self.gt_axises = cat([p.gt_axises for p in proposals], dim=0)
                if "BMOC_V0" not in self.motionnet_type and "BMOC_V1" not in self.motionnet_type:
                    self.gt_extrinsic = cat([p.gt_extrinsic for p in proposals], dim=0)
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

    def mstate_loss(self):
        if self._no_instances:
            return 0.0 * self.pred_mstate.sum()
        else:
            # Use all foreground to calculate the loss, not only motion valid
            bg_class_ind = self.pred_class_logits.shape[1] - 1
            fg_inds = nonzero_tuple(
                (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
            )[0]
            if self.state_bgfg:
                return F.cross_entropy(
                    self.pred_mstate[fg_inds], self.gt_states.long()[fg_inds], reduction="sum"
                ) / self.gt_classes.numel()
            else:
                return F.cross_entropy(
                    self.pred_mstate[fg_inds], self.gt_states.long()[fg_inds], reduction="mean"
                )

    # MotionNet: add new losses for mtype, morigin, maxis
    def mtype_loss(self):
        if self._no_instances:
            return (
                0.0 * self.pred_mtype.sum()
            )  # todo: change binary x entropy to softmax (with two lables)
        # return F.binary_cross_entropy_with_logits(torch.squeeze(self.pred_mtype), self.gt_types, reduction='mean')
        # 4.12 if motion valids == False, then don't calculate the loss
        if self.pred_mtype[self.gt_motion_valids].size(0) == 0:
            return 0.0 * self.pred_mtype.sum()
        else:
            return F.cross_entropy(
                self.pred_mtype[self.gt_motion_valids],
                self.gt_types.long()[self.gt_motion_valids],
                reduction="sum",
            ) / self.valid_total_num

    def morigin_loss(self):
        if self._no_instances:
            return 0.0 * self.pred_morigin.sum()
        # 4.12 if motion valids == False, then don't calculate the loss
        if self.pred_morigin[self.gt_motion_valids].size(0) == 0:
            return 0.0 * self.pred_morigin.sum()
        else:
            rot_inds = (
                (self.gt_types[self.gt_motion_valids] == 0).nonzero().unbind(1)[0]
            )
            if self.pred_morigin[self.gt_motion_valids][rot_inds].size(0) == 0:
                return 0.0 * self.pred_morigin.sum()
            return smooth_l1_loss(
                self.pred_morigin[self.gt_motion_valids][rot_inds],
                self.gt_origins[self.gt_motion_valids][rot_inds],
                self.smooth_l1_beta,
                reduction="sum",
            ) / self.valid_total_num

    # Axis has been normalized
    def maxis_loss(self):
        if self._no_instances:
            return 0.0 * self.pred_maxis.sum()
        # target = torch.ones(self.pred_maxis.size()[0]).cuda()
        # return F.cosine_embedding_loss(self.pred_maxis, self.gt_axises, target, reduction='mean')

        # 2.04 normalize the maxis
        self.pred_maxis = F.normalize(self.pred_maxis, p=2, dim=1)
        # 4.12 if motion valids == False, then don't calculate the loss
        if self.pred_maxis[self.gt_motion_valids].size(0) == 0:
            return 0.0 * self.pred_maxis.sum()
        else:
            return smooth_l1_loss(
                self.pred_maxis[self.gt_motion_valids],
                self.gt_axises[self.gt_motion_valids],
                self.smooth_l1_beta,
                reduction="sum",
            ) / self.valid_total_num

    def extrinsic_loss(self):
        if self._no_instances:
            return 0.0 * self.pred_extrinsic.sum()
        # MSE version
        if self.pred_extrinsic[self.gt_motion_valids].size(0) == 0:
            return 0.0 * self.pred_extrinsic.sum()
        else:
            gt_extrinsic_parameter = torch.cat(
                [
                    self.gt_extrinsic[self.gt_motion_valids][:, 0:3],
                    self.gt_extrinsic[self.gt_motion_valids][:, 4:7],
                    self.gt_extrinsic[self.gt_motion_valids][:, 8:11],
                    self.gt_extrinsic[self.gt_motion_valids][:, 12:15],
                ],
                1,
            )
            return smooth_l1_loss(
                self.pred_extrinsic[self.gt_motion_valids],
                gt_extrinsic_parameter,
                self.smooth_l1_beta,
                reduction="sum",
            ) / self.valid_total_num

    # MotionNet: log the motion accuracy
    @torch.no_grad()
    def _log_motion(self):
        num_instances = self.gt_types.numel()
        # CHOICE
        # pred_probs = torch.sigmoid(torch.squeeze(self.pred_mtype))
        # pred_types = (pred_probs > 0.5).float()
        # num_accurate_type = (pred_types == self.gt_types).nonzero().numel()

        # MotionNet 2.03
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
            loss_cls = torch.tensor(0.0, device=self.gt_classes.device)
            loss_box_reg = torch.tensor(0.0, device=self.gt_classes.device)
        else:
            loss_cls = self.softmax_cross_entropy_loss()
            loss_box_reg = self.box_reg_loss()

        losses = {
            "loss_cls": loss_cls,
            "loss_box_reg": loss_box_reg,
            "loss_mtype": self.mtype_loss(),
            "loss_morigin": self.morigin_loss(),
            "loss_maxis": self.maxis_loss(),
        }

        if "BMOC_V0" not in self.motionnet_type and "BMOC_V1" not in self.motionnet_type:
            losses["loss_extrinsic"] = self.extrinsic_loss()
        
        if self.motionstate:
            losses["loss_mstate"] = self.mstate_loss()
        # import pdb
        # pdb.set_trace()
        return losses

    def predict_boxes(self):
        """
        Deprecated
        """
        return self._predict_boxes().split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        """
        Deprecated
        """
        probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image):
        """
        Deprecated
        """
        boxes = self.predict_boxes()
        scores = self.predict_probs()
        image_shapes = self.image_shapes
        return fast_rcnn_inference(
            boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image
        )


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
        use_GTBBX=False,
        use_GTEXTRINSIC=False,
        use_GTCAT=False,
        freeze_DET=False,
        most_frequent_gt=False,
        most_frequent_type=None,
        most_frequent_origin=None,
        most_frequent_axis=None,
        most_frequent_pred=False,
        origin_NOC=False,
        most_frequent_origin_NOC=None,
        random_NOC=False,
        canAxes_NOC=None,
        canOrigins_NOC=None,
        MODELATTRPATH=None,
        motionnet_type=None,
        motionstate=False,
        state_bgfg=False,
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
        self.motionnet_type = motionnet_type
        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        # self.cls_score = Linear(input_size, num_classes + 1)
        # MotionNet4.13: update the classification error
        self.cls_score = nn.Sequential(
            Linear(input_size, 512),
            nn.ReLU(inplace=True),
            Linear(512, 128),
            nn.ReLU(inplace=True),
            Linear(128, 32),
            nn.ReLU(inplace=True),
            Linear(32, num_classes + 1),
        )
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        # self.bbox_pred = Linear(input_size, num_bbox_reg_classes * box_dim)
        self.bbox_pred = nn.Sequential(
            Linear(input_size, 512),
            nn.ReLU(inplace=True),
            Linear(512, 128),
            nn.ReLU(inplace=True),
            Linear(128, 32),
            nn.ReLU(inplace=True),
            Linear(32, num_bbox_reg_classes * box_dim),
        )

        for layer in self.cls_score:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
                )
        for layer in self.bbox_pred:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
                )

        if motionstate:
            # Currently only consider binary states
            self.mstate_layer = nn.Sequential(
                Linear(input_size, 512),
                nn.ReLU(inplace=True),
                Linear(512, 128),
                nn.ReLU(inplace=True),
                Linear(128, 32),
                nn.ReLU(inplace=True),
                Linear(32, 2),
            )

        # MotionNet
        self.mtype_layer = nn.Sequential(
            Linear(input_size, 512),
            nn.ReLU(inplace=True),
            Linear(512, 128),
            nn.ReLU(inplace=True),
            Linear(128, 32),
            nn.ReLU(inplace=True),
            Linear(32, 2),
        )

        # For motion origin and motion axis, both outputs are three values
        self.morigin_layer = nn.Sequential(
            Linear(input_size, 512),
            nn.ReLU(inplace=True),
            Linear(512, 128),
            nn.ReLU(inplace=True),
            Linear(128, 32),
            nn.ReLU(inplace=True),
            Linear(32, 3),
        )
        self.maxis_layer = nn.Sequential(
            Linear(input_size, 512),
            nn.ReLU(inplace=True),
            Linear(512, 128),
            nn.ReLU(inplace=True),
            Linear(128, 32),
            nn.ReLU(inplace=True),
            Linear(32, 3),
        )
        if "BMOC_V0" not in self.motionnet_type and "BMOC_V1" not in self.motionnet_type:
            self.extrinsic_layer = nn.Sequential(
                Linear(input_size, 512),
                nn.ReLU(inplace=True),
                Linear(512, 128),
                nn.ReLU(inplace=True),
                Linear(128, 32),
                nn.ReLU(inplace=True),
                Linear(32, 12),
            )

            for layer in self.extrinsic_layer:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(
                        layer.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
                    )

        # Init the weights of three new layers for MotionNet
        for layer in self.mtype_layer:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
                )
        for layer in self.morigin_layer:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
                )
        for layer in self.maxis_layer:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
                )

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        self.box_reg_loss_weight = box_reg_loss_weight

        self.use_GTBBX = use_GTBBX
        self.use_GTEXTRINSIC = use_GTEXTRINSIC
        self.use_GTCAT = use_GTCAT
        self.freeze_DET = freeze_DET
        self.most_frequent_gt = most_frequent_gt

        self.most_frequent_type = most_frequent_type
        self.most_frequent_origin = most_frequent_origin
        self.most_frequent_axis = most_frequent_axis
        self.most_frequent_pred = most_frequent_pred

        self.origin_NOC = origin_NOC
        self.most_frequent_origin_NOC = most_frequent_origin_NOC

        self.random_NOC = random_NOC
        self.canAxes_NOC = canAxes_NOC
        self.canOrigins_NOC = canOrigins_NOC

        self.MODELATTRPATH = MODELATTRPATH
        self.motionstate = motionstate

        self.state_bgfg = state_bgfg

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

        if "USE_GTEXTRINSIC" in cfg.MODEL:
            use_GTEXTRINSIC = cfg.MODEL.USE_GTEXTRINSIC
        else:
            use_GTEXTRINSIC = False

        if "FREEZE_DET" in cfg.MODEL:
            freeze_DET = cfg.MODEL.FREEZE_DET
        else:
            freeze_DET = False

        if "MOST_FREQUENT_GT" in cfg.MODEL:
            most_frequent_gt = cfg.MODEL.MOST_FREQUENT_GT
        else:
            most_frequent_gt = False

        if "MOST_FREQUENT_PRED" in cfg.MODEL:
            most_frequent_pred = cfg.MODEL.MOST_FREQUENT_PRED
        else:
            most_frequent_pred = False

        if most_frequent_gt or most_frequent_pred:
            most_frequent_type = cfg.MODEL.most_frequent_type
            most_frequent_origin = cfg.MODEL.most_frequent_origin
            most_frequent_axis = cfg.MODEL.most_frequent_axis
        else:
            most_frequent_type = None
            most_frequent_origin = None
            most_frequent_axis = None

        if "ORIGIN_NOC" in cfg.MODEL:
            origin_NOC = cfg.MODEL.ORIGIN_NOC
        else:
            origin_NOC = False

        if origin_NOC:
            most_frequent_origin_NOC = cfg.MODEL.most_frequent_origin_NOC
        else:
            most_frequent_origin_NOC = None

        if "RANDOM_NOC" in cfg.MODEL:
            random_NOC = cfg.MODEL.RANDOM_NOC
        else:
            random_NOC = False

        if random_NOC:
            canAxes_NOC = cfg.MODEL.canAxes_NOC
            canOrigins_NOC = cfg.MODEL.canOrigins_NOC
        else:
            canAxes_NOC = None
            canOrigins_NOC = None

        if "STATE_BGFG" in cfg.MODEL:
            state_bgfg = cfg.MODEL.STATE_BGFG
        else:
            state_bgfg = False

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
            "use_GTBBX": use_GTBBX,
            "use_GTCAT": use_GTCAT,
            "use_GTEXTRINSIC": use_GTEXTRINSIC,
            "freeze_DET": freeze_DET,
            "most_frequent_gt": most_frequent_gt,
            "most_frequent_type": most_frequent_type,
            "most_frequent_origin": most_frequent_origin,
            "most_frequent_axis": most_frequent_axis,
            "most_frequent_pred": most_frequent_pred,
            "origin_NOC": origin_NOC,
            "most_frequent_origin_NOC": most_frequent_origin_NOC,
            "random_NOC": random_NOC,
            "canAxes_NOC": canAxes_NOC,
            "canOrigins_NOC": canOrigins_NOC,
            "MODELATTRPATH": cfg.MODEL.MODELATTRPATH,
            "motionnet_type": cfg.MODEL.MOTIONNET.TYPE,
            "motionstate": cfg.MODEL.MOTIONSTATE,
            "state_bgfg": state_bgfg
        }

    def getModifiedBound(self):
        model_attr_file = open(self.MODELATTRPATH)
        model_bound = json.load(model_attr_file)
        model_attr_file.close()
        # The data on the urdf need a coordinate transform [x, y, z] -> [z, x, y]
        modified_bound = {}
        for model in model_bound:
            modified_bound[model] = {}
            min_bound = model_bound[model]["min_bound"]
            modified_bound[model]["min_bound"] = np.array(
                [min_bound[2], min_bound[0], min_bound[1]]
            )
            max_bound = model_bound[model]["max_bound"]
            modified_bound[model]["max_bound"] = np.array(
                [max_bound[2], max_bound[0], max_bound[1]]
            )
            modified_bound[model]["scale_factor"] = (
                modified_bound[model]["max_bound"] - modified_bound[model]["min_bound"]
            )
        return modified_bound

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
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        # Predict the value for the motion type, motion origin and motion axis
        if self.freeze_DET:
            x = x.detach()
        mstate = None
        if self.motionstate:
            mstate = self.mstate_layer(x)
        mtype = self.mtype_layer(x)
        morigin = self.morigin_layer(x)
        maxis = self.maxis_layer(x)
        if "BMOC_V0" not in self.motionnet_type and "BMOC_V1" not in self.motionnet_type:
            extrinsic = self.extrinsic_layer(x)
        else:
            extrinsic = None
        
        return scores, proposal_deltas, mtype, morigin, maxis, extrinsic, mstate

    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features
                that were used to compute predictions.
        """
        scores, proposal_deltas, mtype, morigin, maxis, extrinsic, mstate = predictions
        # print(self.maxis_layer.weight.grad)
        return MotionOutputs(
            self.box2box_transform,
            scores,
            proposal_deltas,
            mtype,
            morigin,
            maxis,
            extrinsic,
            mstate,
            proposals,
            self.smooth_l1_beta,
            self.box_reg_loss_type,
            self.box_reg_loss_weight,
            self.freeze_DET,
            self.motionnet_type,
            self.motionstate,
            self.state_bgfg,
        ).losses()

    # MotionNet: functions used when testing
    def inference(self, predictions, proposals, pred_extrinsics=None):
        """
        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        if scores[0].size(0) == 0:
            mtype = (torch.zeros((0)),)
            morigin = (torch.zeros((0, 9)),)
            maxis = (torch.zeros((0, 3)),)
            mextrinsic = (torch.zeros((0, 12)),)
            mstate = (torch.zeros((0)),)
        else:
            mtype, morigin, maxis, mextrinsic, mstate = self.predict_motions(
                predictions, proposals, pred_extrinsics
            )

            morigin = list(morigin)
            maxis = list(maxis)

            for j in range(len(mextrinsic)):
                axis_end = morigin[j] + maxis[j]
                mextrinsic_c2w = torch.eye(4, device=mextrinsic[j].device).repeat(
                    mextrinsic[j].size(0), 1, 1
                )
                mextrinsic_c2w[:, 0:3, 0:4] = torch.transpose(
                    mextrinsic[j].reshape(mextrinsic[j].size(0), 4, 3), 1, 2
                )
                mextrinsic_w2c = torch.inverse(mextrinsic_c2w)
                morigin[j] = (
                    torch.matmul(
                        mextrinsic_w2c[:, :3, :3], morigin[j].unsqueeze(2)
                    ).squeeze(2)
                    + mextrinsic_w2c[:, :3, 3]
                )
                end_in_cam = (
                    torch.matmul(
                        mextrinsic_w2c[:, :3, :3], axis_end.unsqueeze(2)
                    ).squeeze(2)
                    + mextrinsic_w2c[:, :3, 3]
                )
                maxis[j] = end_in_cam - morigin[j]
                # import pdb
                # pdb.set_trace()
                # # mextrinsic_w2c = np.array([np.linalg.inv(np.concatenate([mextrinsic[j][i].cpu().reshape(4, 3), np.array([[0,0,0,1]]).transpose()], axis=1).transpose()) for i in range(mextrinsic[j].size()[0])])
                # morigin[j] = torch.tensor(np.array([np.dot(mextrinsic_w2c[i][0:3, 0:3], morigin[j][i].cpu())+mextrinsic_w2c[i][0:3, 3] for i in range(mextrinsic_w2c.shape[0])]))
                # end_in_cam = torch.tensor(np.array([np.dot(mextrinsic_w2c[i][0:3, 0:3], axis_end[i])+mextrinsic_w2c[i][0:3, 3] for i in range(mextrinsic_w2c.shape[0])]))
                # maxis[j] = end_in_cam - morigin[j]

        image_shapes = [x.image_size for x in proposals]
        return motion_inference(
            boxes,
            scores,
            mtype,
            morigin,
            maxis,
            mextrinsic,
            mstate,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
            self.motionstate,
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
        scores, proposal_deltas, mtype, morigin, maxis, extrinsic, mstate = predictions

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
        scores, proposal_deltas, mtype, morigin, maxis, extrinsic, mstate = predictions
        # If using gtbbx, then make the delta to zero
        if self.use_GTBBX:
            proposal_deltas = torch.zeros(
                proposal_deltas.size(), device=proposal_deltas.device
            )

        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(self, predictions, proposals):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        scores, proposal_deltas, mtype, morigin, maxis, extrinsic, mstate = predictions
        num_inst_per_image = [len(p) for p in proposals]

        if self.random_NOC:
            scores = torch.rand(scores.size(), device=scores.device)

        probs = F.softmax(scores, dim=-1)

        if self.use_GTCAT and not probs.size(0) == 0:
            probs = torch.zeros(probs.size(), device=probs.device)
            gt_classes = cat([p.gt_classes for p in proposals], dim=0)
            probs[range(probs.size(0)), gt_classes] = 1

        return probs.split(num_inst_per_image, dim=0)

    # MotionNet: predict results for the motion information
    def predict_motions(self, predictions, proposals, pred_extrinsics=None):
        scores, proposal_deltas, mtype, morigin, maxis, mextrinsic, mstate = predictions
        num_inst_per_image = [len(p) for p in proposals]

        if "BMOC_V0" in self.motionnet_type or "BMOC_V1" in self.motionnet_type:
            assert mextrinsic == None
            assert not pred_extrinsics == None
            assert len(proposals) == pred_extrinsics.size(0)
            mextrinsic = cat(
                [
                    torch.stack([pred_extrinsic] * num_inst)
                    for pred_extrinsic, num_inst in zip(
                        pred_extrinsics, num_inst_per_image
                    )
                ],
                dim=0,
            )

        assert not mextrinsic == None

        if self.most_frequent_gt:
            most_frequent_type = torch.tensor(
                self.most_frequent_type, device=scores.device
            )
            most_frequent_axis = torch.tensor(
                self.most_frequent_axis, device=scores.device, dtype=torch.float32
            )
            gt_classes = cat([p.gt_classes for p in proposals], dim=0)
            # Process the mtype
            frequent_type = most_frequent_type[gt_classes]
            mtype = torch.zeros(mtype.size(), device=mtype.device, dtype=torch.float32)
            mtype[range(mtype.size(0)), frequent_type] = 1
            # Process the morigin
            if not self.origin_NOC and not self.random_NOC:
                most_frequent_origin = torch.tensor(
                    self.most_frequent_origin, device=scores.device, dtype=torch.float32
                )
                morigin = most_frequent_origin[gt_classes]
                # Process the maxis
                maxis = most_frequent_axis[gt_classes]
            elif self.random_NOC:
                modified_bound = self.getModifiedBound()
                # Pick a random normalized origin and scale it back to the world coordinate
                canOrigins_NOC = torch.tensor(
                    self.canOrigins_NOC, device=scores.device, dtype=torch.float32
                )
                gt_model_name = cat([p.gt_model_name for p in proposals], dim=0)
                min_bound = torch.tensor(
                    [
                        modified_bound[str(model_name)]["min_bound"]
                        for model_name in gt_model_name.cpu().numpy()
                    ],
                    device=mtype.device,
                    dtype=torch.float32,
                )
                scale_factor = torch.tensor(
                    [
                        modified_bound[str(model_name)]["scale_factor"]
                        for model_name in gt_model_name.cpu().numpy()
                    ],
                    device=mtype.device,
                    dtype=torch.float32,
                )
                random_origin_index = torch.randint(0, 19, (gt_classes.size(0),))
                morigin = (
                    canOrigins_NOC[random_origin_index] + 0.5
                ) * scale_factor + min_bound
                # pick a random axis from the candidate axes
                canAxes_NOC = torch.tensor(
                    self.canAxes_NOC, device=scores.device, dtype=torch.float32
                )
                random_axes_index = torch.randint(0, 3, (gt_classes.size(0),))
                maxis = canAxes_NOC[random_axes_index]
            else:
                modified_bound = self.getModifiedBound()
                most_frequent_origin_NOC = torch.tensor(
                    self.most_frequent_origin_NOC,
                    device=scores.device,
                    dtype=torch.float32,
                )
                gt_model_name = cat([p.gt_model_name for p in proposals], dim=0)
                min_bound = torch.tensor(
                    [
                        modified_bound[str(model_name)]["min_bound"]
                        for model_name in gt_model_name.cpu().numpy()
                    ],
                    device=mtype.device,
                    dtype=torch.float32,
                )
                scale_factor = torch.tensor(
                    [
                        modified_bound[str(model_name)]["scale_factor"]
                        for model_name in gt_model_name.cpu().numpy()
                    ],
                    device=mtype.device,
                    dtype=torch.float32,
                )
                morigin = (
                    most_frequent_origin_NOC[gt_classes] + 0.5
                ) * scale_factor + min_bound
                # Process the maxis
                maxis = most_frequent_axis[gt_classes]

        if self.most_frequent_pred:
            try:
                most_frequent_type = torch.tensor(
                    self.most_frequent_type, device=scores.device
                )
                most_frequent_axis = torch.tensor(
                    self.most_frequent_axis, device=scores.device, dtype=torch.float32
                )
                pred_classes = torch.argmax(scores, axis=-1)
                pred_classes[pred_classes > 2] = 2

                # Process the mtype
                if self.random_NOC:
                    frequent_type = torch.randint(0, 2, (pred_classes.size(0),))
                else:
                    frequent_type = most_frequent_type[pred_classes]
                mtype = torch.zeros(
                    mtype.size(), device=mtype.device, dtype=torch.float32
                )
                mtype[range(mtype.size(0)), frequent_type] = 1
                # Process the morigin
                if not self.origin_NOC and not self.random_NOC:
                    most_frequent_origin = torch.tensor(
                        self.most_frequent_origin,
                        device=scores.device,
                        dtype=torch.float32,
                    )
                    morigin = most_frequent_origin[pred_classes]
                    # Process the maxis
                    maxis = most_frequent_axis[pred_classes]
                elif self.random_NOC:
                    modified_bound = self.getModifiedBound()
                    # Pick a random normalized origin and scale it back to the world coordinate
                    canOrigins_NOC = torch.tensor(
                        self.canOrigins_NOC, device=scores.device, dtype=torch.float32
                    )
                    gt_model_name = cat([p.gt_model_name for p in proposals], dim=0)
                    min_bound = torch.tensor(
                        [
                            modified_bound[str(model_name)]["min_bound"]
                            for model_name in gt_model_name.cpu().numpy()
                        ],
                        device=mtype.device,
                        dtype=torch.float32,
                    )
                    scale_factor = torch.tensor(
                        [
                            modified_bound[str(model_name)]["scale_factor"]
                            for model_name in gt_model_name.cpu().numpy()
                        ],
                        device=mtype.device,
                        dtype=torch.float32,
                    )
                    random_origin_index = torch.randint(0, 19, (pred_classes.size(0),))
                    morigin = (
                        canOrigins_NOC[random_origin_index] + 0.5
                    ) * scale_factor + min_bound
                    # pick a random axis from the candidate axes
                    canAxes_NOC = torch.tensor(
                        self.canAxes_NOC, device=scores.device, dtype=torch.float32
                    )
                    random_axes_index = torch.randint(0, 3, (pred_classes.size(0),))
                    maxis = canAxes_NOC[random_axes_index]
                else:
                    modified_bound = self.getModifiedBound()
                    most_frequent_origin_NOC = torch.tensor(
                        self.most_frequent_origin_NOC,
                        device=scores.device,
                        dtype=torch.float32,
                    )
                    gt_model_name = cat([p.gt_model_name for p in proposals], dim=0)
                    min_bound = torch.tensor(
                        [
                            modified_bound[str(model_name)]["min_bound"]
                            for model_name in gt_model_name.cpu().numpy()
                        ],
                        device=mtype.device,
                        dtype=torch.float32,
                    )
                    scale_factor = torch.tensor(
                        [
                            modified_bound[str(model_name)]["scale_factor"]
                            for model_name in gt_model_name.cpu().numpy()
                        ],
                        device=mtype.device,
                        dtype=torch.float32,
                    )
                    morigin = (
                        most_frequent_origin_NOC[pred_classes] + 0.5
                    ) * scale_factor + min_bound
                    # Process the maxis
                    maxis = most_frequent_axis[pred_classes]

            except:
                # Some images when inferencing may not have valid annotations
                pass

        if self.use_GTEXTRINSIC:
            try:
                gt_extrinsic = cat([p.gt_extrinsic for p in proposals], dim=0)
                mextrinsic = torch.cat(
                    [
                        gt_extrinsic[:, 0:3],
                        gt_extrinsic[:, 4:7],
                        gt_extrinsic[:, 8:11],
                        gt_extrinsic[:, 12:15],
                    ],
                    1,
                )
            except:
                # Some images when inferencing may not have valid annotations
                pass

        # CHOICE

        # # Process the mtype to make it be binary
        # mtype = torch.sigmoid(mtype)
        # mtype = (mtype > 0.5).float()

        if self.motionstate:
            pred_probs = F.softmax(mstate, dim=1)
            mstate = torch.argmax(pred_probs, 1).float()

        # MotionNet 2.03
        pred_probs = F.softmax(mtype, dim=1)
        mtype = torch.argmax(pred_probs, 1).float()

        # 2.04 normalize the maxis
        maxis = F.normalize(maxis, p=2, dim=1)

        if self.motionstate:
            return (
                mtype.split(num_inst_per_image, dim=0),
                morigin.split(num_inst_per_image, dim=0),
                maxis.split(num_inst_per_image, dim=0),
                mextrinsic.split(num_inst_per_image, dim=0),
                mstate.split(num_inst_per_image, dim=0),
            )
        else:
            return (
                mtype.split(num_inst_per_image, dim=0),
                morigin.split(num_inst_per_image, dim=0),
                maxis.split(num_inst_per_image, dim=0),
                mextrinsic.split(num_inst_per_image, dim=0),
                None,
            )


# MotionNet: based on fast_rcnn_inference
def motion_inference(
    boxes,
    scores,
    mtype,
    morigin,
    maxis,
    mextrinsic,
    mstate,
    image_shapes,
    score_thresh,
    nms_thresh,
    topk_per_image,
    motionstate,
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
    if motionstate:
        result_per_image = [
            motion_inference_single_image(
                boxes_per_image,
                scores_per_image,
                mtype_per_image,
                morigin_per_image,
                maxis_per_image,
                mextrinsic_per_image,
                mstate_per_image,
                image_shape,
                score_thresh,
                nms_thresh,
                topk_per_image,
                motionstate,
            )
            for scores_per_image, boxes_per_image, mtype_per_image, morigin_per_image, maxis_per_image, mextrinsic_per_image, mstate_per_image, image_shape in zip(
                scores, boxes, mtype, morigin, maxis, mextrinsic, mstate, image_shapes
            )
        ]
    else:
        result_per_image = [
            motion_inference_single_image(
                boxes_per_image,
                scores_per_image,
                mtype_per_image,
                morigin_per_image,
                maxis_per_image,
                mextrinsic_per_image,
                None,
                image_shape,
                score_thresh,
                nms_thresh,
                topk_per_image,
                motionstate,
            )
            for scores_per_image, boxes_per_image, mtype_per_image, morigin_per_image, maxis_per_image, mextrinsic_per_image, image_shape in zip(
                scores, boxes, mtype, morigin, maxis, mextrinsic, image_shapes
            )
        ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


# MotionNet: based on fast_rcnn_inference_single_image
def motion_inference_single_image(
    boxes,
    scores,
    mtype,
    morigin,
    maxis,
    mextrinsic,
    mstate,
    image_shape,
    score_thresh,
    nms_thresh,
    topk_per_image,
    motionstate,
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
        mtype = mtype[valid_mask]
        morigin = morigin[valid_mask]
        maxis = maxis[valid_mask]
        mextrinsic = mextrinsic[valid_mask]
        if motionstate:
            mstate = mstate[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # Filter results based on detection scores
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    mtype = mtype[filter_inds[:, 0]]
    morigin = morigin[filter_inds[:, 0]]
    maxis = maxis[filter_inds[:, 0]]
    mextrinsic = mextrinsic[filter_inds[:, 0]]
    if motionstate:
        mstate = mstate[filter_inds[:, 0]]

    # Apply per-class NMS
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, mtype, morigin, maxis, mextrinsic, filter_inds = (
        boxes[keep],
        scores[keep],
        mtype[keep],
        morigin[keep],
        maxis[keep],
        mextrinsic[keep],
        filter_inds[keep],
    )
    if motionstate:
        mstate = mstate[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.mtype = mtype
    result.morigin = morigin
    result.maxis = maxis
    result.mextrinsic = mextrinsic
    if motionstate:
        result.mstate = mstate
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]
