from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads, build_box_head
from detectron2.config import configurable
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import FastRCNNOutputLayers
from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou

from .bmcc_motion_output import MotionOutputLayers as BMCCOutputLayers
from .bmoc_motion_output import MotionOutputLayers as BMOCOutputLayers
from .pm_motion_output import MotionOutputLayers as PMOutputLayers

import torch
from typing import Dict, List, Optional, Tuple, Union


@ROI_HEADS_REGISTRY.register()
class MotionROIHeads(StandardROIHeads):
    @configurable
    def __init__(
        self,
        # *,
        # box_in_features: List[str],
        # box_pooler: ROIPooler,
        # box_head: nn.Module,
        # box_predictor: nn.Module,
        # mask_in_features: Optional[List[str]] = None,
        # mask_pooler: Optional[ROIPooler] = None,
        # mask_head: Optional[nn.Module] = None,
        # keypoint_in_features: Optional[List[str]] = None,
        # keypoint_pooler: Optional[ROIPooler] = None,
        # keypoint_head: Optional[nn.Module] = None,
        # train_on_pred_boxes: bool = False,
        use_GT = False,
        freeze_DET=False,
        motionnet_type=None,
        use_GTBBX=False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.use_GT = use_GT
        self.freeze_DET = freeze_DET
        self.motionnet_type = motionnet_type
        self.use_GTBBX = use_GTBBX

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)

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

        if "USE_GTPOSE" in cfg.MODEL:
            use_GTPOSE = cfg.MODEL.USE_GTPOSE
        else:
            use_GTPOSE = False

        if "FREEZE_DET" in cfg.MODEL:
            freeze_DET = cfg.MODEL.FREEZE_DET
        else:
            freeze_DET = False

        if "ORIGIN_NOC" in cfg.MODEL:
            origin_NOC = cfg.MODEL.ORIGIN_NOC
        else:
            origin_NOC = False

        if "RANDOM_NOC" in cfg.MODEL:
            random_NOC = cfg.MODEL.RANDOM_NOC
        else:
            random_NOC = False

        # use_GT means that add gt to the inference 
        use_GT = use_GTEXTRINSIC or use_GTPOSE or use_GTCAT or origin_NOC or random_NOC

        ret["use_GT"] = use_GT
        ret["freeze_DET"] = freeze_DET
        ret["motionnet_type"] = cfg.MODEL.MOTIONNET.TYPE
        ret["use_GTBBX"] = use_GTBBX

        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels, height=pooler_resolution, width=pooler_resolution
            ),
        )

        # TODO(AXC): Improve this (dedup code in output layers)        
        output_layers = {
            "BMCC": BMCCOutputLayers,
            "BMOC": BMOCOutputLayers,
            "PM": PMOutputLayers,
            "PM_V0": PMOutputLayers,
            "BMOC_V0": BMOCOutputLayers,
            "BMOC_V1": BMOCOutputLayers,
        }
        box_predictor = output_layers[cfg.MODEL.MOTIONNET.TYPE](cfg, box_head.output_shape)

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    @torch.no_grad()
    def label_and_sample_proposals_gt(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        # if self.proposal_append_gt:
        #     proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)

            # Don't sample the proposals when using gt bbx
            sampled_idxs = torch.tensor(range(matched_idxs.size(0)), device=matched_idxs.device)
            has_gt = targets_per_image.gt_classes.numel() > 0
            # Get the corresponding GT for each proposal
            if has_gt:
                gt_classes = targets_per_image.gt_classes[matched_idxs]
                # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
                gt_classes[matched_labels == 0] = self.num_classes
                # Label ignore proposals (-1 label)
                gt_classes[matched_labels == -1] = -1
            else:
                gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
            # sampled_idxs, gt_classes = self._sample_proposals(
            #     matched_idxs, matched_labels, targets_per_image.gt_classes
            # )


            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        return proposals_with_gt

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        pred_extrinsics = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        # For training, if using GTBBX, then no need to add additional GT or sample proposals, but we need to match them
        if self.training and not self.use_GTBBX:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        elif self.training and self.use_GTBBX:
            assert targets
            # For gtbbx and gtcat, gt bbx may not exist
            if not proposals[0].objectness_logits.size(0) == 0:
                proposals = self.label_and_sample_proposals_gt(proposals, targets)

        # For inferencing, don't want to add GT or sample proposals, but if we need GT information, we need to match them
        if not self.training and self.use_GT:
            assert targets
            # For gtbbx and gtcat, gt bbx may not exist
            if not proposals[0].objectness_logits.size(0) == 0:
                proposals = self.label_and_sample_proposals_gt(proposals, targets)

        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            if not self.freeze_DET:
                losses.update(self._forward_mask(features, proposals))
                losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals, pred_extrinsics)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def _forward_box(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances], pred_extrinsics = None,
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                        proposals, pred_boxes
                    ):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            if "BMOC_V0" in self.motionnet_type or "BMOC_V1" in self.motionnet_type:
                pred_instances, _ = self.box_predictor.inference(predictions, proposals, pred_extrinsics)
            else:
                pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances


