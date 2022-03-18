from detectron2.data import *
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.config import configurable
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)

import copy
import logging
import math
import numpy as np
import json
from PIL import Image
import os
from typing import List, Optional, Union
import torch
import pycocotools.mask as mask_util
import h5py

MOTION_TYPE = {"rotation": 0, "translation": 1}

# MotionNet Version: based on DatasetMapper
class MotionDatasetMapper(DatasetMapper):
    @configurable
    def __init__(
        self,
        # is_train: bool,
        # *,
        # augmentations: List[Union[T.Augmentation, T.Transform]],
        # image_format: str,
        # use_instance_mask: bool = False,
        # use_keypoint: bool = False,
        # instance_mask_format: str = "polygon",
        # keypoint_hflip_indices: Optional[np.ndarray] = None,
        # precomputed_proposal_topk: Optional[int] = None,
        # recompute_boxes: bool = False,
        annotations_to_instances=None,
        network_type=None,
        use_GTBBX=False,
        use_GT=False,
        motionstate=False,
        image_state_path=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.annotations_to_instances = annotations_to_instances
        self.network_type = network_type
        self.use_GTBBX = use_GTBBX
        self.use_GT = use_GT
        self.motionstate = motionstate 
        self.image_state_path = image_state_path

        if self.motionstate:
            if self.image_state_path == None:
                raise ValueError("No image state file")
            with open(self.image_state_path, "r") as f:
                self.image_states = json.load(f)

        self.images = None
        self.filenames = None
        self.filenames_map = {}

        self.depth_images = None
        self.depth_filenames = None
        self.depth_filenames_map = {}

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        if "USE_GTBBX" in cfg.MODEL:
            use_GTBBX = cfg.MODEL.USE_GTBBX
        else:
            use_GTBBX = False

        # TODO(AXC): we should use config to get augmentation vs having this be built in here
        #            then we can remove custom version of code
        # augs = utils.build_augmentation(cfg, is_train)
        # Add augmentation for training
        augs = []

        if is_train:
            if not use_GTBBX:
                augs.append(T.RandomFlip(0.5))
            else:
                augs.append(T.NoOpTransform())
            augs.append(T.RandomBrightness(0.5, 1.5))
            augs.append(T.RandomContrast(0.5, 1.5))

        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False


        def f(
            annos, shape, mask_format, motion_valid, extrinsic_matrix, model_name
        ):
            return bm_annotations_to_instances(
                annos,
                shape,
                mask_format=mask_format,
                motion_valid=motion_valid,
                extrinsic_matrix=extrinsic_matrix,
                model_name=model_name,
                motionstate=cfg.MODEL.MOTIONSTATE,
            )

        annotations_to_instances = f

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

        # use_GT means that add gt to the inference
        use_GT = use_GTEXTRINSIC or use_GTCAT or origin_NOC or random_NOC

        if "IMAGESTATEPATH" in cfg.MODEL:
            image_state_path = cfg.MODEL.IMAGESTATEPATH
        else:
            image_state_path = None

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
            "annotations_to_instances": annotations_to_instances,
            "network_type": cfg.MODEL.MOTIONNET.TYPE,
            "use_GTBBX": use_GTBBX,
            "use_GT": use_GT,
            "motionstate": cfg.MODEL.MOTIONSTATE,
            "image_state_path": image_state_path,
        }
        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(
                cfg.DATASETS.TRAIN
            )

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret

    def add_GT_into_proposals(self, gt_boxes, image_size):
        # Assign all ground-truth boxes an objectness logit corresponding to
        # P(object) = sigmoid(logit) =~ 1.
        gt_logit_value = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))
        gt_logits = gt_logit_value * torch.ones(len(gt_boxes))

        # Concatenating gt_boxes with proposals requires them to have the same fields
        gt_proposal = Instances(image_size)
        gt_proposal.proposal_boxes = gt_boxes
        gt_proposal.objectness_logits = gt_logits

        # gt_proposals = Instances.cat([gt_proposal, gt_proposal])
        # # Gerneate 1000 proposals
        # for i in range(998):
        #     gt_proposals = Instances.cat([gt_proposals, gt_proposal])

        return gt_proposal

    def load_depth_h5(self, base_dir):
        # Load the dataset at the first time
        if self.depth_images == None:
            depth_h5file = h5py.File(f'{base_dir}/depth.h5')
            self.depth_images = depth_h5file['depth_images']
            self.depth_filenames = depth_h5file['depth_filenames']
            num_images = self.depth_filenames.shape[0]
            for i in range(num_images):
                self.depth_filenames_map[self.depth_filenames[i].decode('utf-8')] = i

    def load_h5(self, base_dir, dir):
        if self.images == None:
            h5file = h5py.File(f'{base_dir}/{dir}.h5')
            self.images = h5file[f'{dir}_images']
            self.filenames = h5file[f'{dir}_filenames']
            num_images = self.filenames.shape[0]
            for i in range(num_images):
                self.filenames_map[self.filenames[i].decode('utf-8')] = i

    
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # MotionNet: load images from the h5 file
        # Get the Dataset path
        base_dir = os.path.split(os.path.split(dataset_dict["file_name"])[0])[0]
        dir = os.path.split(os.path.split(dataset_dict["file_name"])[0])[-1]
        file_name = os.path.split(dataset_dict["file_name"])[-1]
        
        if self.image_format == "depth":
            self.load_depth_h5(base_dir)
            image = self.depth_images[self.depth_filenames_map[dataset_dict["depth_file_name"]]]
        elif self.image_format == "RGB":
            self.load_h5(base_dir, dir)
            image = self.images[self.filenames_map[file_name]]
        elif self.image_format == "RGBD":
            self.load_depth_h5(base_dir)
            self.load_h5(base_dir, dir)
            depth_image = self.depth_images[self.depth_filenames_map[dataset_dict["depth_file_name"]]]
            RGB_image = self.images[self.filenames_map[file_name]]
            image = np.concatenate([RGB_image, depth_image], axis=2)
        

        # #### MotionNet: support different image format (Load the data from image files)
        # if self.image_format == "depth":
        #     base_dir = os.path.split(os.path.split(dataset_dict["file_name"])[0])[0]
        #     image = np.array(
        #         Image.open(f'{base_dir}/depth/{dataset_dict["depth_file_name"]}'),
        #         dtype=np.float32,
        #     )[:, :, None]
        # elif self.image_format == "RGB":
        #     image = utils.read_image(
        #         dataset_dict["file_name"], format=self.image_format
        #     )
        # elif self.image_format == "RGBD":
        #     base_dir = os.path.split(os.path.split(dataset_dict["file_name"])[0])[0]
        #     RGB_image = utils.read_image(dataset_dict["file_name"], format="RGB")
        #     depth_image = np.array(
        #         Image.open(f'{base_dir}/depth/{dataset_dict["depth_file_name"]}'),
        #         dtype=np.float32,
        #     )[:, :, None]
        #     image = np.concatenate([RGB_image, depth_image], axis=2)
        # ####

        model_name = dataset_dict["file_name"].split("/")[-1].split("-")[0]

        utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image, sem_seg=None)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        #### MotionNet: apply transform on image and mask/bbx; if there is flip operation, then set motion_valid to false
        motion_valid = False
        if self.is_train:
            # Judge if there is no random flip, then motion annotations are valid
            if isinstance(transforms[0], T.NoOpTransform):
                motion_valid = True
        else:
            # When inferencing, all motions are valid; currently inferencing code doesn't use the attribute
            motion_valid = True
        #####

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        #### MotionNet
        # Only use gt in world coordinate for BMOC. For other cases, no need for extra transformation
        if "extrinsic" in dataset_dict["camera"] and "BMOC" in self.network_type:
            extrinsic_matrix = np.array(dataset_dict["camera"]["extrinsic"]["matrix"])
        else:
            extrinsic_matrix = None

        # All other annotations are in camera coordinate, assume the camera intrinsic parameters are fixed (Just used in final visualization)
        dataset_dict.pop("camera")
        dataset_dict.pop("depth_file_name")
        dataset_dict.pop("label")
        ####

        if not self.is_train and self.use_GTBBX == False and self.use_GT == False:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            if self.motionstate:
                rgb_name = dataset_dict["file_name"].split("/")[-1].split(".")[0]
                part_state = self.image_states[rgb_name]
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)
                if self.motionstate:
                    part_id = anno["motion"]["partId"]
                    anno["motion"]["motionStateBin"] = int(part_state[part_id]["close"])
                    anno["motion"]["motionStateCon"] = part_state[part_id]["value"]

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]

            #### MotionNet
            # Convert the annotations into tensor
            # Add motion valid to instance to indicate if the instance will be used for motion loss
            instances = self.annotations_to_instances(
                annos,
                image_shape,
                mask_format=self.instance_mask_format,
                motion_valid=motion_valid,
                extrinsic_matrix=extrinsic_matrix,
                model_name=model_name,
            )
            ####

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

            # Add the GTBBX as the proposals
            if self.use_GTBBX:
                proposals = self.add_GT_into_proposals(instances.gt_boxes, image_shape)
                dataset_dict["proposals"] = proposals

        return dataset_dict


# MotionNet: add motion type, motion origin. motion axis
# For motion valid = False, extrinsic matrix and motion parameters may not be the true gt
# This is to better train the detection and segmentation
def bm_annotations_to_instances(
    annos, image_size, mask_format="polygon", motion_valid=True, extrinsic_matrix=None, model_name=None, motionstate=False
):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = [
        BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS)
        for obj in annos
    ]
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    target.gt_model_name = torch.tensor(
            [int(model_name)] * len(annos), dtype=torch.int32
        )

    # Add the motionStateBin if it's in the annotations
    if motionstate == True:
        states = [obj["motion"]["motionStateBin"] for obj in annos]
        states = torch.tensor(states, dtype=torch.float32)
        target.gt_states = states

    # Add extra annotations: gt_origins, gt_axises and gt_types
    types = [MOTION_TYPE[obj["motion"]["type"]] for obj in annos]
    types = torch.tensor(types, dtype=torch.float32)
    target.gt_types = types

    if extrinsic_matrix is not None:
        extrinsic_matrix = torch.tensor(
            [extrinsic_matrix] * len(annos), dtype=torch.float32
        )
        target.gt_extrinsic = extrinsic_matrix

        if len(annos) == 0:
            transformation = None
            target.gt_extrinsic_rxs_cat = torch.tensor([])
            target.gt_extrinsic_rxs_residual = torch.tensor([])
            target.gt_extrinsic_rxs_cover = torch.tensor([])

            target.gt_extrinsic_rys_cat = torch.tensor([])
            target.gt_extrinsic_rys_residual = torch.tensor([])
            target.gt_extrinsic_rys_cover = torch.tensor([])

            target.gt_extrinsic_rzs_cat = torch.tensor([])
            target.gt_extrinsic_rzs_residual = torch.tensor([])
            target.gt_extrinsic_rzs_cover = torch.tensor([])
        else:
            origin_extrinsic = np.zeros(16)
            origin_extrinsic[-1] = 1
            origin_extrinsic[0:3] = extrinsic_matrix[0][0:3]
            origin_extrinsic[4:7] = extrinsic_matrix[0][4:7]
            origin_extrinsic[8:11] = extrinsic_matrix[0][8:11]
            origin_extrinsic[12:15] = extrinsic_matrix[0][12:15]
            transformation = np.reshape(origin_extrinsic, (4, 4)).T
    else:
        transformation = None

    origins_cam = [obj["motion"]["current_origin"] for obj in annos]
    if transformation is not None:
        origins_world = [
            np.dot(transformation, np.array(origin[:] + [1])) for origin in origins_cam
        ]
        origins_world = np.asarray(origins_world)[:, 0:3]
        origins = torch.tensor(origins_world, dtype=torch.float32)
    else:
        origins = torch.tensor(origins_cam, dtype=torch.float32)
    target.gt_origins = origins

    axes_cam = [obj["motion"]["current_axis"] for obj in annos]
    if transformation is not None:
        axes_end_cam = list(np.asarray(axes_cam) + np.asarray(origins_cam))
        axes_end_world = [
            np.dot(transformation, np.append(axis, 1)) for axis in axes_end_cam
        ]
        axes_end_world = np.asarray(axes_end_world)[:, 0:3]
        axes_world = axes_end_world - origins_world
        axes = torch.tensor(axes_world, dtype=torch.float32)
    else:
        axes = torch.tensor(axes_cam, dtype=torch.float32)
    target.gt_axises = axes

    motion_valids = [motion_valid] * len(annos)
    target.gt_motion_valids = torch.tensor(motion_valids)

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        if mask_format == "polygon":
            # TODO check type and provide better error
            masks = PolygonMasks(segms)
        else:
            assert mask_format == "bitmask", mask_format
            masks = []
            for segm in segms:
                if isinstance(segm, list):
                    # polygon
                    masks.append(polygons_to_bitmask(segm, *image_size))
                elif isinstance(segm, dict):
                    # COCO RLE
                    masks.append(mask_util.decode(segm))
                elif isinstance(segm, np.ndarray):
                    assert (
                        segm.ndim == 2
                    ), "Expect segmentation of 2 dimensions, got {}.".format(segm.ndim)
                    # mask array
                    masks.append(segm)
                else:
                    raise ValueError(
                        "Cannot convert segmentation of type '{}' to BitMasks!"
                        "Supported types are: polygons as list[list[float] or ndarray],"
                        " COCO-style RLE as a dict, or a full-image segmentation mask "
                        "as a 2D ndarray.".format(type(segm))
                    )
            # torch.from_numpy does not support array with negative stride.
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
            )
        target.gt_masks = masks

    if len(annos) and "keypoints" in annos[0]:
        kpts = [obj.get("keypoints", []) for obj in annos]
        target.gt_keypoints = Keypoints(kpts)

    return target

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The reuslt is for euler angles (ZYX) radians
def rotationMatrixToEulerAngles(R):

    assert isRotationMatrix(R)

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

def batchEulerAnglesToRotationMatrix(theta):
    R_x = np.tile(np.eye(3), [theta.shape[0], 1, 1])
    R_x[:, 1, 1] = np.cos(theta[:, 0])
    R_x[:, 1, 2] = -np.sin(theta[:, 0])
    R_x[:, 2, 1] = np.sin(theta[:, 0])
    R_x[:, 2, 2] = np.cos(theta[:, 0])
    R_y = np.tile(np.eye(3), [theta.shape[0], 1, 1])
    R_y[:, 0, 0] = np.cos(theta[:, 1])
    R_y[:, 0, 2] = np.sin(theta[:, 1])
    R_y[:, 2, 0] = -np.sin(theta[:, 1])
    R_y[:, 2, 2] = np.cos(theta[:, 1])
    R_z = np.tile(np.eye(3), [theta.shape[0], 1, 1])
    R_z[:, 0, 0] = np.cos(theta[:, 2])
    R_z[:, 0, 1] = -np.sin(theta[:, 2])
    R_z[:, 1, 0] = np.sin(theta[:, 2])
    R_z[:, 1, 1] = np.cos(theta[:, 2])
    rotation_matrixs = np.matmul(R_z, np.matmul(R_y, R_x))

    return rotation_matrixs

