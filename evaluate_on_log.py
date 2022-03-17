import argparse
import datetime
import os
import torch
import json
import numpy as np
from time import time

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from motlib import MotionTrainer, add_motionnet_config, register_motion_instances


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_motionnet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.TEST.EVAL_PERIOD = 0
    # Store the most frequent parameters, in the order
    # Current use the statistics in train
    # 0: drawer, 1: door, 2: lid
    cfg.MODEL.most_frequent_type = [1, 0, 0]
    # Below most_frequent_origin is deprecated
    cfg.MODEL.most_frequent_origin = [
        [0, 0, 0],
        [0, 0, 0],
        [-0.41908362, 0.53434741, 0.48216539],
    ]
    # Below is MF statistics on our synthetic data
    cfg.MODEL.most_frequent_origin_NOC = [
        [0.0, 0.0, 0.0],
        [-0.5, 0.0, -0.5],
        [0.0, 0.5, 0.5],
    ]
    cfg.MODEL.most_frequent_axis = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]

    # Below is MF statistics on our real data
    # cfg.MODEL.most_frequent_origin_NOC = [
    #     [0.0, 0.0, -0.5],
    #     [0.5, 0.5, 0.0],
    #     [0.5, 0.0, 0.0],
    # ]
    # cfg.MODEL.most_frequent_axis = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]

    # The candidate axis and origins for BM-OC-MF-NOC-RM
    cfg.MODEL.canAxes_NOC = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    cfg.MODEL.canOrigins_NOC = [
        [0, 0, 0],
        [0.5, 0, -0.5],
        [0.5, 0, 0.5],
        [-0.5, 0, -0.5],
        [-0.5, 0, 0.5],
        [0, 0.5, 0.5],
        [0, 0.5, -0.5],
        [0, -0.5, 0.5],
        [0, -0.5, -0.5],
        [0.5, 0.5, 0],
        [0.5, -0.5, 0],
        [-0.5, 0.5, 0],
        [-0.5, -0.5, 0],
        [0, 0.5, 0],
        [0, -0.5, 0],
        [0.5, 0, 0],
        [-0.5, 0, 0],
        [0, 0, 0.5],
        [0, 0, -0.5],
    ]

    # Output directory
    cfg.OUTPUT_DIR = args.output_dir

    # Inference file
    cfg.INFERENCE_FILE = args.inference_file
    # Filter file
    cfg.FILTER_FILE = args.filter_file

    # Input format
    # TODO: maybe read from file
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

    cfg.MODEL.AxisThres = args.motion_threshold[0]
    cfg.MODEL.OriginThres = args.motion_threshold[1]

    # Option for testing
    cfg.MODEL.MOST_FREQUENT_GT = args.most_frequent_gt
    if args.most_frequent_gt:
        args.gtbbx = True
        args.gtcat = True
        args.gtextrinsic = True

    cfg.MODEL.MOST_FREQUENT_PRED = args.most_frequent_pred

    # Below two options is only for MOST_FREQUENT_PRED case
    cfg.MODEL.ORIGIN_NOC = args.origin_NOC
    cfg.MODEL.RANDOM_NOC = args.random_NOC

    # if not cfg.MODEL.MOTIONNET.TYPE == "BMOC" and (
    #     cfg.MODEL.MOST_FREQUENT_GT
    #     or cfg.MODEL.MOST_FREQUENT_PRED
    #     or cfg.MODEL.ORIGIN_NOC
    #     or cfg.MODEL.RANDOM_NOC
    # ):
    #     raise ValueError("Invalid most frequent option for not BMOC model")

    cfg.MODEL.USE_GTBBX = args.gtbbx
    if args.gtbbx:
        cfg.MODEL.PROPOSAL_GENERATOR.NAME = "PrecomputedProposals"
        # The below setting will only influence the training process
        # cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT = False

    cfg.MODEL.USE_GTCAT = args.gtcat

    cfg.MODEL.USE_GTEXTRINSIC = args.gtextrinsic
    # if not cfg.MODEL.MOTIONNET.TYPE == "BMOC" and cfg.MODEL.USE_GTEXTRINSIC:
    #     raise ValueError("Invalid extrinsic option for not BMOC model")

    cfg.MODEL.USE_GTPOSE = args.gtpose
    if not cfg.MODEL.MOTIONNET.TYPE == "PM" and cfg.MODEL.USE_GTPOSE:
        raise ValueError("Invalid pose option for not PM model")

    cfg.MODEL.MODELATTRPATH = args.model_attr_path
    if cfg.MODEL.MOTIONSTATE:
        cfg.MODEL.IMAGESTATEPATH = args.image_state_path

    cfg.MODEL.RANDOM_BASELINE = args.random_baseline
    if cfg.MODEL.RANDOM_BASELINE and (
        cfg.MODEL.MOST_FREQUENT_GT or cfg.MODEL.MOST_FREQUENT_PRED
    ):
        raise ValueError(
            "Invalid: use the random baseline and most frequent baseline at the same time"
        )

    if not cfg.MODEL.MOTIONNET.TYPE == "BMCC" and cfg.MODEL.RANDOM_BASELINE:
        raise ValueError("Random baseline currently works for BMCC")

    cfg.MODEL.TYPE_MATCH = args.type_match
    cfg.MODEL.PART_CAT = args.part_cat
    cfg.MODEL.MICRO_AVG = args.micro_avg

    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Evaluate motion net")
    parser.add_argument(
        "--config-file",
        default="configs/bmcc.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--output-dir",
        default=f"eval_output/{datetime.datetime.now().isoformat()}",
        metavar="DIR",
        help="path for evaluation output",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        metavar="DIR",
        help="path containing motion datasets",
    )
    parser.add_argument(
        "--inference-file",
        default=None,
        metavar="FILE",
        help="path to the inference file. If this value is not None, then the program will use existing predictions instead of inferencing again",
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
        "--motion_threshold",
        nargs=2,
        type=float,
        default=[10, 0.25],
        help="the threshold for axis and origin for calculating mAP",
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
    # Deprecated gtpose -> not design corresponding ablation yet
    parser.add_argument(
        "--gtpose",
        action="store_true",
        help="indicating whether to use GT pose for pm",
    )
    # The below option will automatically load some sub option (For most fr)
    # most_frequent_gt will automatically set gtbbx, gtcat & gtextrinsic to be true
    parser.add_argument(
        "--most_frequent_gt",
        action="store_true",
        help="indicating whether to choose most frequent baseline with gtbbx, gtcat & gtextrinsic. This option can only be used with BM-OC",
    )
    # most_frequent_pred will only use the most frequent motion type/origin/axis
    parser.add_argument(
        "--most_frequent_pred",
        action="store_true",
        help="indicating whether to choose most frequent baselien with predicted bbx, part label, extrinsic. This option can only be used with BM-OC",
    )
    # most_frequent_pred origin will be in the normalized object coordinate if this is set to true
    parser.add_argument(
        "--origin_NOC",
        action="store_true",
        help="indicating whether the most frequent origin is in NOC (normalized object coordinate). This option can only be used with BM-OC",
    )
    parser.add_argument(
        "--random_NOC",
        action="store_true",
        help="Randomly pick axis and origin from the most_frequent_candidates. This option can only be used with BM-OC",
    )
    parser.add_argument(
        "--model_attr_path",
        required=False,
        default="/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/data_statistics/urdf-attr.json",
        help="indicating the path to ",
    )
    parser.add_argument(
        "--image_state_path",
        # default="/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/motion_state/image_close.json",
        help="indicating the path to part states for each image -> used to train and evaluate",
    )
    # Deprecated random_baseline, not we use most_frequent_pred and random_NOC for the random baseline
    parser.add_argument(
        "--random_baseline",
        action="store_true",
        help="Assign random number to the motion axis, motion origin and motion type. This option can only be used with BM-CC",
    )
    # The below option are for special evaluation metric
    parser.add_argument(
        "--type_match",
        action="store_true",
        help="indicating whether the evaluation metric for unmatched motion type is needed",
    )
    parser.add_argument(
        "--part_cat",
        action="store_true",
        help="indicating whether the evaluation metric is for each part category (e.g. drawer, door, lid)",
    )
    parser.add_argument(
        "--micro_avg",
        action="store_true",
        help="indicating whether micro-average is applied (statistics on all examples without considering part categories)",
    )
    # Additional file list used for the evaluation to use only part of the val set
    parser.add_argument(
        "--filter-file",
        default=None,
        metavar="FILE",
        help="path to the filter file which includes part of the valid images",
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

    with torch.no_grad():
        trainer = MotionTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.test(trainer.cfg, trainer.model)

    stop = time()
    print(str(stop - start) + " seconds")