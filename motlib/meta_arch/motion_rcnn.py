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
        freeze_DET=False,
        motionnet_type=None,
        smooth_l1_beta=0.0,
        extrinsic_weight=1,
        motion_weights=[1, 1, 1],
        pose_weights=[1, 1, 1],
        pose_rt_weight=[1, 1],
        mstate_weight=1,
        det_extrinsic_iter=0,
        ROTATION_BIN_NUM=None,
        COVER_VALUE=None,
        use_GTBBX=False,
        use_GTCAT=False,
        use_GTEXTRINSIC=False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.use_GT = use_GT
        self.freeze_DET = freeze_DET

        # This is for single extrinsic matrix for BMOC, hardcode to use 'p2' feature
        self.motionnet_type = motionnet_type
        self.smooth_l1_beta = smooth_l1_beta
        self.extrinsic_weight = extrinsic_weight
        self.motion_weights = motion_weights
        self.pose_weights = pose_weights
        self.pose_rt_weight = pose_rt_weight
        self.mstate_weight = mstate_weight
        self.det_extrinsic_iter = det_extrinsic_iter
        self.ROTATION_BIN_NUM = ROTATION_BIN_NUM
        self.rotation_bin = getRotationBin(ROTATION_BIN_NUM, COVER_VALUE)
        self.use_GTBBX = use_GTBBX
        self.use_GTCAT = use_GTCAT
        self.use_GTEXTRINSIC = use_GTEXTRINSIC
        
        if "BMOC_V0" in self.motionnet_type or "BMOC_V1" in self.motionnet_type:
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

        if "BMOC_V1" in self.motionnet_type:
            assert not ROTATION_BIN_NUM == None
            self.extrinsic_trans_layer = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 3), # 16 * 3
            )
            for layer in self.extrinsic_trans_layer:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(
                        layer.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
                    )
                    nn.init.constant_(layer.bias, 0)
            self.extrinsic_rot_feature_layer = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
            )
            for layer in self.extrinsic_rot_feature_layer:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(
                        layer.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
                    )
                    nn.init.constant_(layer.bias, 0)
            self.extrinsic_cat_layers = nn.ModuleList()
            self.extrinsic_residual_layers = nn.ModuleList()
            for i in range(3):
                self.extrinsic_cat_layers.append(nn.Sequential(
                    nn.Linear(512, 128),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(128, 32),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(32, ROTATION_BIN_NUM),
                ))
                for layer in self.extrinsic_cat_layers[i]:
                    if isinstance(layer, nn.Linear):
                        nn.init.kaiming_normal_(
                            layer.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
                        )
                        nn.init.constant_(layer.bias, 0)
                self.extrinsic_residual_layers.append(nn.Sequential(
                    nn.Linear(512, 128),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(128, 32),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(32, 2 * ROTATION_BIN_NUM),
                ))
                for layer in self.extrinsic_residual_layers[i]:
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

        if "EXTRINSIC_WEIGHT" in cfg.MODEL:
            extrinsic_weight = cfg.MODEL.EXTRINSIC_WEIGHT
        else:
            extrinsic_weight = 1

        if "MOTION_WEIGHTS" in cfg.MODEL:
            motion_weights = cfg.MODEL.MOTION_WEIGHTS
        else:
            motion_weights = [1, 1, 1]

        if "POSE_WEIGHTS" in cfg.MODEL:
            pose_weights = cfg.MODEL.POSE_WEIGHTS
        else:
            pose_weights = [1, 1, 1]

        if "POSE_RT_WEIGHT" in cfg.MODEL:
            pose_rt_weight = cfg.MODEL.POSE_RT_WEIGHT
        else:
            pose_rt_weight = [1, 1]

        if "MSTATE_WEIGHT" in cfg.MODEL:
            mstate_weight = cfg.MODEL.MSTATE_WEIGHT
        else:
            mstate_weight = 1

        if "DET_EXTRINSIC_ITER" in cfg.MODEL:
            det_extrinsic_iter = cfg.MODEL.DET_EXTRINSIC_ITER
        else:
            det_extrinsic_iter = 0

        if "ROTATION_BIN_NUM" in cfg.INPUT:
            ROTATION_BIN_NUM = cfg.INPUT.ROTATION_BIN_NUM
        else:
            ROTATION_BIN_NUM = None

        if "COVER_VALUE" in cfg.INPUT:
            COVER_VALUE = cfg.INPUT.COVER_VALUE
        else:
            COVER_VALUE = None


        # use_GT means that add gt to the inference 
        use_GT = use_GTEXTRINSIC or use_GTPOSE or use_GTCAT or origin_NOC or random_NOC

        ret["use_GT"] = use_GT
        ret["freeze_DET"] = freeze_DET
        ret["motionnet_type"] = cfg.MODEL.MOTIONNET.TYPE
        ret["smooth_l1_beta"] = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA
        ret["extrinsic_weight"] = extrinsic_weight
        ret["motion_weights"] = motion_weights
        ret["pose_weights"] = pose_weights
        ret["pose_rt_weight"] = pose_rt_weight
        ret["mstate_weight"] = mstate_weight
        ret["det_extrinsic_iter"] = det_extrinsic_iter
        ret["ROTATION_BIN_NUM"] = ROTATION_BIN_NUM
        ret["COVER_VALUE"] = COVER_VALUE
        
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

        if "BMOC_V0" in self.motionnet_type or "BMOC_V1" in self.motionnet_type:
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
        if "BMOC_V1" in self.motionnet_type:
            # Get the pred extrinsic trans 
            pred_extrinsic_trans = self.extrinsic_trans_layer(extrinsic_feature)
            rot_feature = self.extrinsic_rot_feature_layer(extrinsic_feature)
            pred_rotx_cat = self.extrinsic_cat_layers[0](rot_feature)
            pred_rotx_residual = self.extrinsic_residual_layers[0](rot_feature)
            pred_roty_cat = self.extrinsic_cat_layers[1](rot_feature)
            pred_roty_residual = self.extrinsic_residual_layers[1](rot_feature)
            pred_rotz_cat = self.extrinsic_cat_layers[2](rot_feature)
            pred_rotz_residual = self.extrinsic_residual_layers[2](rot_feature)
            # Calculate the loss with the gt
            extrinsic_losses = {"loss_extrinsic": torch.tensor(0.0, dtype=torch.float32, device=pred_rotx_cat.device)}
            num_instance = 0
            for i in range(len(gt_instances)):
                gt_extrinsic_rxs_cat = gt_instances[i].gt_extrinsic_rxs_cat
                gt_extrinsic_rxs_residual = gt_instances[i].gt_extrinsic_rxs_residual
                gt_extrinsic_rxs_cover = gt_instances[i].gt_extrinsic_rxs_cover

                gt_extrinsic_rys_cat = gt_instances[i].gt_extrinsic_rys_cat
                gt_extrinsic_rys_residual = gt_instances[i].gt_extrinsic_rys_residual
                gt_extrinsic_rys_cover = gt_instances[i].gt_extrinsic_rys_cover

                gt_extrinsic_rzs_cat = gt_instances[i].gt_extrinsic_rzs_cat
                gt_extrinsic_rzs_residual = gt_instances[i].gt_extrinsic_rzs_residual
                gt_extrinsic_rzs_cover = gt_instances[i].gt_extrinsic_rzs_cover

                gt_extrinsic_trans = gt_instances[i].gt_extrinsic_trans

                # # Below is for debugging using gt
                # import pdb
                # pdb.set_trace()
                # pred_extrinsic_trans[i] = gt_instances[i].gt_extrinsic_trans[0].unsqueeze(0)
                # pred_rotx_cat[i] = F.one_hot(gt_instances[i].gt_extrinsic_rxs_cat[0].long(), num_classes=self.ROTATION_BIN_NUM).unsqueeze(0)
                # pred_rotx_residual[i] = gt_instances[i].gt_extrinsic_rxs_residual[0].unsqueeze(0)
                # pred_roty_cat[i] = F.one_hot(gt_instances[i].gt_extrinsic_rys_cat[0].long(), num_classes=self.ROTATION_BIN_NUM).unsqueeze(0)
                # pred_roty_residual[i] = gt_instances[i].gt_extrinsic_rys_residual[0].unsqueeze(0)
                # pred_rotz_cat[i] = F.one_hot(gt_instances[i].gt_extrinsic_rzs_cat[0].long(), num_classes=self.ROTATION_BIN_NUM).unsqueeze(0)
                # pred_rotz_residual[i] = gt_instances[i].gt_extrinsic_rzs_residual[0].unsqueeze(0)
                
                if gt_extrinsic_rxs_cat.size(0) == 0:
                    extrinsic_losses["loss_extrinsic"] += 0.0 * pred_rotx_cat[i].sum()
                else:
                    if gt_instances[i].gt_motion_valids[0] == False:
                        extrinsic_losses["loss_extrinsic"] += 0.0 * pred_rotx_cat[i].sum()
                    else:
                        num_instance += 1
                        # Calculate the loss for extrinsic rot x
                        loss = F.cross_entropy(
                            pred_rotx_cat[i].unsqueeze(0),
                            gt_extrinsic_rxs_cat[0].unsqueeze(0).long(),
                            reduction="mean",
                        )
                        ## Normalize; Multiply 2 is for cos and sin
                        normalize_rotx_residual = torch.zeros(
                            pred_rotx_residual[i].size(),
                            device=pred_rotx_residual.device,
                        )
                        for j in range(self.ROTATION_BIN_NUM):
                            temp = (
                                pred_rotx_residual[i][2 * j] ** 2
                                + pred_rotx_residual[i][2 * j + 1] ** 2
                            ) ** 0.5
                            normalize_rotx_residual[2 * j] = (
                                pred_rotx_residual[i][2 * j] / temp
                            )
                            normalize_rotx_residual[2 * j + 1] = (
                                pred_rotx_residual[i][2 * j + 1] / temp
                            )
                        loss += 2 * smooth_l1_loss(
                            normalize_rotx_residual[gt_extrinsic_rxs_cover[0]],
                            gt_extrinsic_rxs_residual[0][
                                gt_extrinsic_rxs_cover[0]
                            ],
                            self.smooth_l1_beta,
                            reduction="mean",
                        )
                        # Calculate the loss for extrinsic rot y
                        loss += F.cross_entropy(
                            pred_roty_cat[i].unsqueeze(0),
                            gt_extrinsic_rys_cat[0].unsqueeze(0).long(),
                            reduction="mean",
                        )
                        ## Normalize; Multiply 2 is for cos and sin
                        normalize_roty_residual = torch.zeros(
                            pred_roty_residual[i].size(),
                            device=pred_roty_residual.device,
                        )
                        for j in range(self.ROTATION_BIN_NUM):
                            temp = (
                                pred_roty_residual[i][2 * j] ** 2
                                + pred_roty_residual[i][2 * j + 1] ** 2
                            ) ** 0.5
                            normalize_roty_residual[2 * j] = (
                                pred_roty_residual[i][2 * j] / temp
                            )
                            normalize_roty_residual[2 * j + 1] = (
                                pred_roty_residual[i][2 * j + 1] / temp
                            )
                        loss += 2 * smooth_l1_loss(
                            normalize_roty_residual[gt_extrinsic_rys_cover[0]],
                            gt_extrinsic_rys_residual[0][
                                gt_extrinsic_rys_cover[0]
                            ],
                            self.smooth_l1_beta,
                            reduction="mean",
                        )
                        # Calculate the loss for extrinsic rot z
                        loss += F.cross_entropy(
                            pred_rotz_cat[i].unsqueeze(0),
                            gt_extrinsic_rzs_cat[0].unsqueeze(0).long(),
                            reduction="mean",
                        )
                        ## Normalize; Multiply 2 is for cos and sin
                        normalize_rotz_residual = torch.zeros(
                            pred_rotz_residual[i].size(),
                            device=pred_rotz_residual.device,
                        )
                        for j in range(self.ROTATION_BIN_NUM):
                            temp = (
                                pred_rotz_residual[i][2 * j] ** 2
                                + pred_rotz_residual[i][2 * j + 1] ** 2
                            ) ** 0.5
                            normalize_rotz_residual[2 * j] = (
                                pred_rotz_residual[i][2 * j] / temp
                            )
                            normalize_rotz_residual[2 * j + 1] = (
                                pred_rotz_residual[i][2 * j + 1] / temp
                            )
                        loss += 2 * smooth_l1_loss(
                            normalize_rotz_residual[gt_extrinsic_rzs_cover[0]],
                            gt_extrinsic_rzs_residual[0][
                                gt_extrinsic_rzs_cover[0]
                            ],
                            self.smooth_l1_beta,
                            reduction="mean",
                        )
                        extrinsic_losses["loss_extrinsic"] += loss / 3
                        # Calculate the extrinsic_trans loss
                        extrinsic_losses["loss_extrinsic"] += smooth_l1_loss(
                            pred_extrinsic_trans[i].unsqueeze(0),
                            gt_extrinsic_trans[0].unsqueeze(0),
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
        if storage.iter >= self.det_extrinsic_iter:
            losses.update(detector_losses)
        else:
            losses.update({'loss_cls': detector_losses['loss_cls'], 'loss_box_reg': detector_losses['loss_box_reg'], 'loss_mask': detector_losses['loss_mask']})
        if not self.freeze_DET:
            losses.update(proposal_losses)
        if "BMOC_V0" in self.motionnet_type or "BMOC_V1" in self.motionnet_type:
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
        # Pose weight (only for PM)
        if "loss_pdimension" in losses.keys() and "loss_ptrans" in losses.keys():
            losses["loss_pdimension"] *= self.pose_weights[0]
        if "loss_ptrans" in losses.keys():
            losses["loss_ptrans"] *= self.pose_weights[1]
        if "loss_prot" in losses.keys():
            losses["loss_prot"] *= self.pose_weights[2]
        # Pose weight (for PM_V0)
        if "loss_pdimension" in losses.keys() and "loss_pose_rt" in losses.keys():
            losses["loss_pdimension"] *= self.pose_rt_weight[0]
        if "loss_pose_rt" in losses.keys():
            losses["loss_pose_rt"] *= self.pose_rt_weight[1]
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
        elif "BMOC_V1" in self.motionnet_type:
            extrinsic_feature = self.extrinsic_feature_layer(features['p2'])
            pred_extrinsic_trans = self.extrinsic_trans_layer(extrinsic_feature)
            rot_feature = self.extrinsic_rot_feature_layer(extrinsic_feature)
            pred_rotx_cat = self.extrinsic_cat_layers[0](rot_feature)
            pred_rotx_residual = self.extrinsic_residual_layers[0](rot_feature)
            pred_roty_cat = self.extrinsic_cat_layers[1](rot_feature)
            pred_roty_residual = self.extrinsic_residual_layers[1](rot_feature)
            pred_rotz_cat = self.extrinsic_cat_layers[2](rot_feature)
            pred_rotz_residual = self.extrinsic_residual_layers[2](rot_feature)
            # # Below is for debugging using gt
            # import pdb
            # pdb.set_trace()
            # pred_extrinsic_trans = gt_instances[0].gt_extrinsic_trans[0].unsqueeze(0)
            # pred_rotx_cat = F.one_hot(gt_instances[0].gt_extrinsic_rxs_cat[0].long(), num_classes=self.ROTATION_BIN_NUM).unsqueeze(0)
            # pred_rotx_residual = gt_instances[0].gt_extrinsic_rxs_residual[0].unsqueeze(0)
            # pred_roty_cat = F.one_hot(gt_instances[0].gt_extrinsic_rys_cat[0].long(), num_classes=self.ROTATION_BIN_NUM).unsqueeze(0)
            # pred_roty_residual = gt_instances[0].gt_extrinsic_rys_residual[0].unsqueeze(0)
            # pred_rotz_cat = F.one_hot(gt_instances[0].gt_extrinsic_rzs_cat[0].long(), num_classes=self.ROTATION_BIN_NUM).unsqueeze(0)
            # pred_rotz_residual = gt_instances[0].gt_extrinsic_rzs_residual[0].unsqueeze(0)
            # Use the prediction to get the extrinsic matrix
            device = pred_rotx_cat.device
            rotation_bin = torch.tensor(self.rotation_bin, device=device)
            theta = torch.zeros((pred_rotx_cat.size(0), 3), device=device,)
            pred_x_cat = torch.argmax(pred_rotx_cat, axis=1)
            normalize_rotx_residual = torch.zeros(
                pred_rotx_residual.size(), device=device,
            )
            for j in range(self.ROTATION_BIN_NUM):
                temp = (
                    pred_rotx_residual[:, 2 * j] ** 2
                    + pred_rotx_residual[:, 2 * j + 1] ** 2
                ) ** 0.5
                normalize_rotx_residual[:, 2 * j] = (
                    pred_rotx_residual[:, 2 * j] / temp
                )
                normalize_rotx_residual[:, 2 * j + 1] = (
                    pred_rotx_residual[:, 2 * j + 1] / temp
                )
            x_residual = torch.acos(
                normalize_rotx_residual[range(pred_x_cat.size(0)), pred_x_cat * 2]
            )
            x_neg_sin_inds = (
                (
                    normalize_rotx_residual[
                        range(pred_x_cat.size(0)), pred_x_cat * 2 + 1
                    ]
                    < 0
                )
                .nonzero()
                .unbind(1)[0]
            )
            x_negative = torch.ones(normalize_rotx_residual.size(0), device=device,)
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
            pred_y_cat = pred_roty_cat.argmax(1)
            normalize_roty_residual = torch.zeros(
                pred_roty_residual.size(), device=device,
            )
            for i in range(self.ROTATION_BIN_NUM):
                temp = (
                    pred_roty_residual[:, 2 * i] ** 2 + pred_roty_residual[:, 2 * i + 1] ** 2
                ) ** 0.5
                normalize_roty_residual[:, 2 * i] = pred_roty_residual[:, 2 * i] / temp
                normalize_roty_residual[:, 2 * i + 1] = (
                    pred_roty_residual[:, 2 * i + 1] / temp
                )
            y_residual = torch.acos(
                normalize_roty_residual[range(pred_y_cat.size(0)), pred_y_cat * 2]
            )
            y_neg_sin_inds = (
                (
                    normalize_roty_residual[
                        range(pred_y_cat.size(0)), pred_y_cat * 2 + 1
                    ]
                    < 0
                )
                .nonzero()
                .unbind(1)[0]
            )
            y_negative = torch.ones(normalize_roty_residual.size(0), device=device,)
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
            pred_z_cat = pred_rotz_cat.argmax(1)
            ## Normalize; Multiply 2 is for cos and sin
            normalize_rotz_residual = torch.zeros(
                pred_rotz_residual.size(), device=device,
            )
            for i in range(self.ROTATION_BIN_NUM):
                temp = (
                    pred_rotz_residual[:, 2 * i] ** 2 + pred_rotz_residual[:, 2 * i + 1] ** 2
                ) ** 0.5
                normalize_rotz_residual[:, 2 * i] = pred_rotz_residual[:, 2 * i] / temp
                normalize_rotz_residual[:, 2 * i + 1] = (
                    pred_rotz_residual[:, 2 * i + 1] / temp
                )
            z_residual = torch.acos(
                normalize_rotz_residual[range(pred_z_cat.size(0)), pred_z_cat * 2]
            )
            z_neg_sin_inds = (
                (
                    normalize_rotz_residual[
                        range(pred_z_cat.size(0)), pred_z_cat * 2 + 1
                    ]
                    < 0
                )
                .nonzero()
                .unbind(1)[0]
            )
            z_negative = torch.ones(normalize_rotz_residual.size(0), device=device,)
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

            pred_extrinsics_matrix = torch.eye(4).repeat(theta.size(0), 1, 1)
            pred_extrinsics_matrix = pred_extrinsics_matrix.to(device)
            pred_extrinsics_matrix[:, :3, :3] = rotation_matrix
            pred_extrinsics_matrix[:, :3, 3] = pred_extrinsic_trans

            pred_extrinsics = torch.flatten(pred_extrinsics_matrix[:, :3, :].transpose(1, 2), 1)
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

def getRotationBin(ROTATION_BIN_NUM, COVER_VALUE):
    bin_range = (2 * np.pi + (ROTATION_BIN_NUM - 1) * COVER_VALUE) / ROTATION_BIN_NUM
    rotation_bin = []
    current_value = -np.pi
    for i in range(ROTATION_BIN_NUM):
        rotation_bin.append([current_value, current_value + bin_range])
        current_value = current_value + bin_range - COVER_VALUE
    rotation_bin[ROTATION_BIN_NUM - 1][1] = np.pi
    return rotation_bin