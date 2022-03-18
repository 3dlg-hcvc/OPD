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
    # Below is MOSTFREQ statistics on our synthetic data
    cfg.MODEL.most_frequent_origin_NOC = [
        [0.0, 0.0, 0.0],
        [-0.5, 0.0, -0.5],
        [0.0, 0.5, 0.5],
    ]
    cfg.MODEL.most_frequent_axis = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]

    # Below is MOSTFREQ statistics on our real data
    # cfg.MODEL.most_frequent_origin_NOC = [
    #     [0.0, 0.0, -0.5],
    #     [0.5, 0.5, 0.0],
    #     [0.5, 0.0, 0.0],
    # ]
    # cfg.MODEL.most_frequent_axis = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]

    # The candidate axis and origins for RANDMOT
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

    # Option for evaluation
    cfg.MODEL.MOST_FREQUENT_PRED = args.most_frequent_pred

    # Below two options is only for MOST_FREQUENT_PRED case
    cfg.MODEL.ORIGIN_NOC = args.origin_NOC
    cfg.MODEL.RANDOM_NOC = args.random_NOC

    cfg.MODEL.USE_GTBBX = args.gtbbx
    if args.gtbbx:
        cfg.MODEL.PROPOSAL_GENERATOR.NAME = "PrecomputedProposals"

    cfg.MODEL.USE_GTCAT = args.gtcat

    cfg.MODEL.USE_GTEXTRINSIC = args.gtextrinsic

    cfg.MODEL.MODELATTRPATH = args.model_attr_path
    if cfg.MODEL.MOTIONSTATE:
        cfg.MODEL.IMAGESTATEPATH = args.image_state_path

    cfg.MODEL.PART_CAT = args.part_cat

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

    # most_frequent_pred will only use the most frequent motion type/origin/axis
    parser.add_argument(
        "--most_frequent_pred",
        action="store_true",
        help="indicating whether to choose most frequent baselien with predicted bbx, extrinsic. This option can only be used with OPDRCNN-O",
    )
    # most_frequent_pred origin will be in the normalized object coordinate if this is set to true
    parser.add_argument(
        "--origin_NOC",
        action="store_true",
        help="indicating whether the most frequent origin is in NOC (normalized object coordinate). This option can only be used with OPDRCNN-O",
    )
    parser.add_argument(
        "--random_NOC",
        action="store_true",
        help="Randomly pick axis and origin from the most_frequent_candidates. This option can only be used with OPDRCNN-O",
    )
    parser.add_argument(
        "--model_attr_path",
        required=True,
        help="indicating the path to ",
    )
    parser.add_argument(
        "--image_state_path",
        help="indicating the path to part states for each image -> used to train and evaluate",
    )
    # The below option are for special evaluation metric
    parser.add_argument(
        "--part_cat",
        action="store_true",
        help="indicating whether the evaluation metric is for each part category (e.g. drawer, door, lid)",
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