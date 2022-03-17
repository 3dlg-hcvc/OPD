import pdb
from detectron2 import model_zoo
from detectron2.data import detection_utils as utils
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model

import os
from time import time
import io
import cv2
import torch
import argparse
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import norm

from PIL import Image

from motlib import (
    MotionTrainer,
    add_motionnet_config,
    register_motion_instances,
    MotionVisualizer,
)
image_path = "/local-scratch/localhome/hja40/Downloads/356.png"

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_motionnet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Output directory
    cfg.OUTPUT_DIR = args.output_dir

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

    cfg.MODEL.MODELATTRPATH = args.model_attr_path

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
        "--input-format",
        default="RGB",
        choices=["RGB", "RGBD", "depth"],
        help="input format (RGB, RGBD, or depth)",
    )
    parser.add_argument(
        "--model_attr_path",
        required=False,
        default="/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/data_statistics/urdf-attr.json",
        help="indicating the path to ",
    )
    parser.add_argument(
        "--prob",
        required=False,
        type=float,
        default=0.5,
        help="indicating the smallest probability to visualize",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
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
    dataset_key = cfg.DATASETS.TEST[0]
    motion_metadata = MetadataCatalog.get(dataset_key)
    dataset_dicts = DatasetCatalog.get(dataset_key)
    
    # Process a Specific Image and get the results
    with torch.no_grad():
        image = Image.open(image_path)
        image = image.resize((256, 256))
        im = utils.convert_PIL_to_numpy(image, "RGB")
        img = torch.as_tensor(
                np.ascontiguousarray(im.transpose(2, 0, 1))
            )
        inputs = {"image": img, "height": 256, "width": 256}

        # Load the weights in to the model
        model = build_model(cfg)
        model.eval()
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        outputs = model([inputs])[0]

    # import pdb
    # pdb.set_trace()
    # todo: the axis currently is not correct, need to change this function to make that correct
    instance_number = len(outputs['instances'])
    count = 0
    for i in range(instance_number):
        v = MotionVisualizer(im,
                    metadata=motion_metadata,
                    scale=2,
                    instance_mode=ColorMode.IMAGE   # remove the colors of unsegmented pixels
        )
        if v.draw_pred_only(outputs["instances"][i].to("cpu"), args.prob) is None:
            continue
        out_filename = os.path.splitext(os.path.basename(image_path))[0]+f'_{count}.png'
        cv2.imwrite(os.path.join(cfg.OUTPUT_DIR, out_filename), v.get_output().get_image()[:, :, ::-1])
        count += 1
        cv2.imshow('', v.get_output().get_image()[:, :, ::-1])
        cv2.waitKey(0)

    stop = time()
    print(str(stop - start) + " seconds")