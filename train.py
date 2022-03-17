import argparse
import datetime
import os

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.utils.env import seed_all_rng
from time import time
import numpy as np

from motlib import MotionTrainer, add_motionnet_config, register_motion_instances


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_motionnet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.TEST.EVAL_PERIOD = 5000
    cfg.merge_from_list(args.opts)
    # Set all random seed to the seed we ask, the default RNG_SEED is set in add_motionnet_config
    if "RNG_SEED" in cfg.INPUT:
        cfg.SEED = cfg.INPUT.RNG_SEED
    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED)
    cfg.INPUT.NOAUG = args.debug_random
    cfg.MODEL.EXTRINSIC_WEIGHT = args.extrinsic_weight
    cfg.MODEL.MOTION_WEIGHTS = args.motion_weights
    cfg.MODEL.POSE_WEIGHTS = args.pose_weights
    cfg.MODEL.POSE_RT_WEIGHT = args.pose_rt_weight
    cfg.MODEL.MSTATE_WEIGHT = args.mstate_weight
    cfg.MODEL.DET_EXTRINSIC_ITER = args.det_extrinsic_iter
    cfg.INPUT.FLIP_PROB = args.flip_prob
    
    cfg.MODEL.STATE_BGFG = args.state_bgfg

    if cfg.MODEL.POSE_ITER > cfg.MODEL.MOTION_ITER and cfg.MODEL.MOTIONNET.TYPE == "PM":
        raise ValueError("For PM, the pose iter should be smaller than the motion iter")

    # Output directory
    cfg.OUTPUT_DIR = args.output_dir

    # Input format
    cfg.INPUT.FORMAT = args.input_format
    if args.input_format == "RGB":
        cfg.MODEL.PIXEL_MEAN = cfg.MODEL.PIXEL_MEAN[0:3]
        cfg.MODEL.PIXEL_STD = cfg.MODEL.PIXEL_STD[0:3]
    elif args.input_format == "depth":
        cfg.MODEL.PIXEL_MEAN = cfg.MODEL.PIXEL_MEAN[3:4]
        cfg.MODEL.PIXEL_STD = cfg.MODEL.PIXEL_STD[3:4]
    elif args.input_format == "RGBD":
        pass
        # cfg.MODEL.PIXEL_MEAN = mean_RGB + mean_depth
        # cfg.MODEL.PIXEL_STD = std_RGB + std_depth
    else:
        raise ValueError("Invalid input format")

    cfg.MODEL.CTPM = args.ctpm

    cfg.MODEL.ONLY_DET = args.only_det
    if not cfg.MODEL.MOTIONNET.TYPE == "BMCC" and cfg.MODEL.ONLY_DET:
        raise ValueError("Invalid only_det option for not BMCC model")

    cfg.MODEL.FREEZE_DET = args.freeze_det
    if cfg.MODEL.FREEZE_DET and cfg.MODEL.ONLY_DET:
        raise ValueError("Cannot use option only_det and freeze_det in the meanwhile")

    if cfg.MODEL.FREEZE_DET and cfg.MODEL.MOTIONNET.TYPE == "PM_V0":
        raise ValueError("Not support yet")

    cfg.MODEL.MODELATTRPATH = args.model_attr_path
    if cfg.MODEL.MOTIONSTATE:
        cfg.MODEL.IMAGESTATEPATH = args.image_state_path

    # Options for ablation study
    cfg.MODEL.USE_GTBBX = args.gtbbx
    if args.gtbbx:
        # Below code is to use gt proposals instead of predicted proposals
        cfg.MODEL.PROPOSAL_GENERATOR.NAME = "PrecomputedProposals"
        # Change the images per batch bigger, because using gt bbx as proposal will make the number of proposals decrease a lot
        cfg.SOLVER.IMS_PER_BATCH = 128

    cfg.MODEL.USE_GTCAT = args.gtcat

    cfg.MODEL.USE_GTEXTRINSIC = args.gtextrinsic

    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Train motion net")
    parser.add_argument(
        "--config-file",
        default="configs/bmcc.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--output-dir",
        default=f"train_output/{datetime.datetime.now().isoformat()}",
        metavar="DIR",
        help="path for training output",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        metavar="DIR",
        help="path containing motion datasets",
    )
    parser.add_argument(
        "--input-format",
        default="RGB",
        choices=["RGB", "RGBD", "depth"],
        help="input format (RGB, RGBD, or depth)",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--ctpm",
        action="store_true",
        help="indicating whether to continue training PM, in this case, use residual with the predicted value",
    )
    parser.add_argument(
        "--only_det",
        action="store_true",
        help="indicating whether to only train the detection part",
    )
    parser.add_argument(
        "--freeze_det",
        action="store_true",
        help="indicating whether to freeze the parameters of the detection part for the models",
    )
    parser.add_argument(
        "--model_attr_path",
        required=True,
        # default="/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/data_statistics/urdf-attr.json",
        help="indicating the path to the diagonal length of each model -> calculate the origin error",
    )
    parser.add_argument(
        "--image_state_path",
        # default="/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/motion_state/image_close.json",
        help="indicating the path to part states for each image -> used to train and evaluate",
    )
    # The below settings are for different weight loss
    parser.add_argument(
        "--extrinsic_weight",
        type=int,
        default=1,
        help="indicating the weights for the extrinsic loss",
    )
    parser.add_argument(
        "--motion_weights",
        nargs=3,
        type=int,
        default=[1, 1, 1],
        help="the weight for [motion_type, motion_axis, motion_origin",
    )
    parser.add_argument(
        "--pose_weights",
        nargs=3,
        type=int,
        default=[1, 1, 1],
        help="the weight for [pose_dimension, pose_trans, pose_rotation",
    )
    parser.add_argument(
        "--pose_rt_weight",
        nargs=2,
        type=int,
        default=[1, 1],
        help="the weight for pose_dimension pose_rt for PM_V0",
    )
    parser.add_argument(
        "--mstate_weight",
        type=int,
        default=1,
        help="the weight for loss_mstate",
    )
    # The below settings are for debugging or new experiments
    parser.add_argument(
        "--det_extrinsic_iter",
        type=int,
        default=0,
        help="indicating whether to only train det and extrinsic for BMOC_V0 ahnd BMOC_V1",
    )
    parser.add_argument(
        "--debug_random",
        action="store_true",
        help="indicating whether to comment all data augmentation during training",
    )
    parser.add_argument(
        "--flip_prob",
        type=float,
        default=0.5,
        help="indicating the probability for random fliping during data augmentation for training",
    )
    # Option for ablation study
    parser.add_argument(
        "--gtbbx",
        action="store_true",
        help="indicating whether to use GT bbx as proposals",
    )
    parser.add_argument(
        "--gtcat",
        action="store_true",
        help="indicating whether to use GT part category",
    )
    parser.add_argument(
        "--gtextrinsic",
        action="store_true",
        help="indicating whether to use GT extrinsic for bmoc",
    )
    # Option for motion state 
    parser.add_argument(
        "--state_bgfg",
        action="store_true",
        help="indicating whether motion state loss is on both fg and bg; default: only on fg",
    )
    return parser


def register_datasets(data_path, cfg):
    dataset_keys = cfg.DATASETS.TRAIN + cfg.DATASETS.TEST
    for dataset_key in dataset_keys:
        json = f"{data_path}/annotations/{dataset_key}.json"
        imgs = f"{data_path}/{dataset_key.split('_')[-1]}"
        register_motion_instances(dataset_key, {}, json, imgs)


# from https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
if __name__ == "__main__":
    start = time()

    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    register_datasets(args.data_path, cfg)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = MotionTrainer(cfg)
    trainer.resume_or_load(resume=False)

    # import pdb
    # pdb.set_trace()
    # # trainer.checkpointer.model

    # trainer.checkpointer.model.backbone.bottom_up.stem.conv1.weight
    trainer.train()

    stop = time()
    print(str(stop - start) + " seconds")
