from numpy.random import random
from detectron2.modeling import GeneralizedRCNN, META_ARCH_REGISTRY
from detectron2.config import configurable
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
from fvcore.nn import smooth_l1_loss
import torch.nn.functional as F
import numpy as np

@META_ARCH_REGISTRY.register()
class MotionRCNN(GeneralizedRCNN):
    @configurable
    def __init__(
        self,
        # *,
        # backbone: Backbone,
        # proposal_generator: nn.Module,
        # roi_heads: nn.Module,
        # pixel_mean: Tuple[float],
        # pixel_std: Tuple[float],
        # input_format: Optional[str] = None,
        # vis_period: int = 0,
        use_GT = False,
        motionnet_type=None,
        smooth_l1_beta=0.0,
        extrinsic_weight=1,
        motion_weights=[1, 1, 1],
        mstate_weight=1,
        use_GTBBX=False,
        use_GTCAT=False,
        use_GTEXTRINSIC=False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.use_GT = use_GT

        # This is for single extrinsic matrix for BMOC, hardcode to use 'p2' feature
        self.motionnet_type = motionnet_type
        self.smooth_l1_beta = smooth_l1_beta
        self.extrinsic_weight = extrinsic_weight
        self.motion_weights = motion_weights
        self.mstate_weight = mstate_weight
        self.use_GTBBX = use_GTBBX
        self.use_GTCAT = use_GTCAT
        self.use_GTEXTRINSIC = use_GTEXTRINSIC
        
        if "BMOC_V0" in self.motionnet_type:
            self.extrinsic_feature_layer = nn.Sequential(
                # 16 * 256 * 64 * 64
                nn.Conv2d(256, 256, 3, 2, 1), # 16 * 256 * 32 * 32
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),  
                nn.MaxPool2d(2, 2), # 16 * 256 * 16 * 16
                nn.Conv2d(256, 256, 3, 2, 1), # 16 * 256 * 8 * 8
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),  
                nn.MaxPool2d(2, 2), # 16 * 256 * 4 * 4
                nn.Conv2d(256, 64, 1), # 16 * 64 * 4 * 4
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),  
                nn.Flatten() # 16 * 1024
            )
            for layer in self.extrinsic_feature_layer:
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        layer.weight, mode="fan_out", nonlinearity="relu"
                    )

        if "BMOC_V0" in self.motionnet_type:
            self.extrinsic_pred_layer = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 12), # 16 * 12
            )

            for layer in self.extrinsic_pred_layer:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(
                        layer.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
                    )
                    nn.init.constant_(layer.bias, 0)

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)

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

        if "ORIGIN_NOC" in cfg.MODEL:
            origin_NOC = cfg.MODEL.ORIGIN_NOC
        else:
            origin_NOC = False

        if "RANDOM_NOC" in cfg.MODEL:
            random_NOC = cfg.MODEL.RANDOM_NOC
        else:
            random_NOC = False

        if "EXTRINSIC_WEIGHT" in cfg.MODEL:
            extrinsic_weight = cfg.MODEL.EXTRINSIC_WEIGHT
        else:
            extrinsic_weight = 1

        if "MOTION_WEIGHTS" in cfg.MODEL:
            motion_weights = cfg.MODEL.MOTION_WEIGHTS
        else:
            motion_weights = [1, 1, 1]

        if "MSTATE_WEIGHT" in cfg.MODEL:
            mstate_weight = cfg.MODEL.MSTATE_WEIGHT
        else:
            mstate_weight = 1

        # use_GT means that add gt to the inference 
        use_GT = use_GTEXTRINSIC or use_GTCAT or origin_NOC or random_NOC

        ret["use_GT"] = use_GT
        ret["motionnet_type"] = cfg.MODEL.MOTIONNET.TYPE
        ret["smooth_l1_beta"] = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA
        ret["extrinsic_weight"] = extrinsic_weight
        ret["motion_weights"] = motion_weights
        ret["mstate_weight"] = mstate_weight
        
        ret["use_GTBBX"] = use_GTBBX
        ret["use_GTCAT"] = use_GTCAT
        ret["use_GTEXTRINSIC"] = use_GTEXTRINSIC

        return ret

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if "BMOC_V0" in self.motionnet_type:
            extrinsic_feature = self.extrinsic_feature_layer(features['p2'])
        if "BMOC_V0" in self.motionnet_type:
            pred_extrinsics = self.extrinsic_pred_layer(extrinsic_feature)
            assert pred_extrinsics.size(0) == len(gt_instances) 
            # calculate the loss of the extrinsic
            extrinsic_losses = {"loss_extrinsic": torch.tensor(0.0, dtype=torch.float32, device=pred_extrinsics.device)}
            num_instance = 0
            for i in range(len(gt_instances)):
                gt_extrinsics = gt_instances[i].gt_extrinsic
                if gt_extrinsics.size(0) == 0:
                    extrinsic_losses["loss_extrinsic"] += 0.0 * pred_extrinsics[i].sum()
                else:
                    if gt_instances[i].gt_motion_valids[0] == False:
                        extrinsic_losses["loss_extrinsic"] += 0.0 * pred_extrinsics[i].sum()
                    else:
                        num_instance += 1
                        gt_extrinsic = gt_extrinsics[0]
                        gt_extrinsic_parameter = torch.cat(
                            [
                                gt_extrinsic[0:3],
                                gt_extrinsic[4:7],
                                gt_extrinsic[8:11],
                                gt_extrinsic[12:15],
                            ],
                            0,
                        )
                        extrinsic_losses["loss_extrinsic"] += smooth_l1_loss(
                                                pred_extrinsics[i],
                                                gt_extrinsic_parameter,
                                                self.smooth_l1_beta,
                                                reduction="mean",
                                            )
            if not num_instance == 0:
                extrinsic_losses["loss_extrinsic"] /= num_instance
        
        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        storage = get_event_storage()
        if self.vis_period > 0:
            # storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        if "BMOC_V0" in self.motionnet_type:
            losses.update(extrinsic_losses)

        if self.use_GTBBX:
            # There is no loss for the RPN
            assert len(list(proposal_losses.keys())) == 0
            if "loss_box_reg" in losses.keys():
                losses.pop("loss_box_reg")
            if "loss_mask" in losses.keys():
                losses.pop("loss_mask")
        if self.use_GTCAT:
            if "loss_cls" in losses.keys():
                losses.pop("loss_cls")
        if self.use_GTEXTRINSIC:
            if "loss_extrinsic" in losses.keys():
                losses.pop("loss_extrinsic")

        # Add the weight to the loss
        # Motion weight (For ALL model)
        if "loss_mtype" in losses.keys():
            losses["loss_mtype"] *= self.motion_weights[0]
        if "loss_maxis" in losses.keys():
            losses["loss_maxis"] *= self.motion_weights[1]
        if "loss_morigin" in losses.keys():
            losses["loss_morigin"] *= self.motion_weights[2]
        # Extrinsic weight (for OC, OC_V0, OC_V1)
        if "loss_extrinsic" in losses.keys():
            losses["loss_extrinsic"] *= self.extrinsic_weight
        if "loss_mstate" in losses.keys():
            losses["loss_mstate"] *= self.mstate_weight
        return losses

    # Modify Inference to enable evaluation using gt extrinsic for oc model
    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        # For using gt, add gt 
        if self.use_GT and "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        features = self.backbone(images.tensor)

        if "BMOC_V0" in self.motionnet_type:
            extrinsic_feature = self.extrinsic_feature_layer(features['p2'])
            pred_extrinsics = self.extrinsic_pred_layer(extrinsic_feature)
        else:
            pred_extrinsics = None

        if self.proposal_generator is not None:
            proposals, _ = self.proposal_generator(images, features, None)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]

        if self.use_GT:
            results, _ = self.roi_heads(images, features, proposals, gt_instances, pred_extrinsics)
        else:
            results, _ = self.roi_heads(images, features, proposals, None, pred_extrinsics)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results
