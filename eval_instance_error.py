from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.utils.logger import setup_logger

import os
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

from motlib import (
    MotionTrainer,
    add_motionnet_config,
    register_motion_instances,
    MotionVisualizer,
)

MOTION_TYPE = {0: "rotation", 1: "translation"}

def getDiagLength(model_attr_path):
    model_attr_file = open(model_attr_path)
    model_bound = json.load(model_attr_file)
    model_attr_file.close()
    # The data on the urdf need a coordinate transform [x, y, z] -> [z, x, y]
    diagonal_length = {}
    for model in model_bound:
        diagonal_length[model] = model_bound[model]["diameter"]
    return diagonal_length

def get_iou(bb1, bb2):
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[0] + bb1[2], bb2[0] + bb2[2])
    y_bottom = min(bb1[1] + bb1[3], bb2[1] + bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = bb1[2] * bb1[3]
    bb2_area = bb2[2] * bb2[3]
    iou = area / float(bb1_area + bb2_area - area)
    return iou


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_motionnet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Output directory
    cfg.OUTPUT_DIR = args.output_dir

    cfg.INFERENCE_FILE = args.inference_file
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Train motion net")
    parser.add_argument(
        "--config-file",
        default="configs/bmcc.yaml",
        metavar="FILE",
        help="Doesn't matter in this file, if we has had the inference file",
    )
    parser.add_argument(
        "--input-format",
        default="RGB",
        metavar="FILE",
        help="Doesn't matter in this file, if we has had the inference file",
    )
    parser.add_argument(
        "--output-dir",
        default=f"/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/analysis/instance_bucket_results",
        metavar="DIR",
        help="path for training output",
    )
    parser.add_argument(
        "--data-path",
        default="/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/MotionDataset_h5_6.11",
        metavar="DIR",
        help="path containing motion datasets",
    )
    parser.add_argument(
        "--inference-file",
        required=True,
        metavar="FILE",
        help="input binary inference file",
    )
    parser.add_argument(
        "--score-threshold",
        default=0.05,
        type=float,
        metavar="VALUE",
        help="a number between 0 and 1. Only consider the predictions whose scores are over the threshold",
    )
    parser.add_argument(
        "--model_attr_path",
        required=False,
        default="/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/data_statistics/urdf-attr.json",
        help="indicating the path to the file which stores the models attributes",
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


if __name__ == "__main__":
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    register_datasets(args.data_path, cfg)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # os.makedirs(f"{cfg.OUTPUT_DIR}/{args.model_name}", exist_ok=True)

    # metadata is just used for the category infomation
    dataset_key = cfg.DATASETS.TEST[0]
    motion_metadata = MetadataCatalog.get(dataset_key)

    # predictor = DefaultPredictor(cfg)
    dataset_dicts = DatasetCatalog.get(dataset_key)

    from fvcore.common.file_io import PathManager

    with PathManager.open(cfg.INFERENCE_FILE, "rb") as f:
        buffer = io.BytesIO(f.read())
    predictions = torch.load(buffer)

    pred_instance_eval = []

    # Draw the valid image
    T_val = [0.5]
    K_val = [0, 1] # here is for motion type category
    A_val = [1e5 ** 2]
    M_val = [100]
    T = len(T_val)
    K = len(K_val)

    diagonal_lengths = getDiagLength(args.model_attr_path)

    count = 0
    total_num_pred = 0
    total_num_rot = 0
    total_num_trans = 0
    for d in dataset_dicts:
        img_path = d["file_name"]
        model_name = (os.path.basename(img_path)).split("-")[0]
        diagonal_length = diagonal_lengths[model_name]

        try:
            prediction = predictions[d["image_id"] - 1]
        except Exception as e:
            raise d["image_id"] - 1

        # im is loaded by cv2 in BGR, Visualizer needs RGB format
        # M_val[0] = min(instance_number, 100)
        pred = prediction["instances"]
        ptind = np.argsort([-p["score"] for p in pred], kind="mergesort")
        pred = [pred[i] for i in ptind[0 : M_val[0]]]

        # Ignore the predictions whose scores are below the score threshold
        temp_pred = pred
        pred = []
        for temp in temp_pred:
            if temp["score"] < args.score_threshold:
                continue
            else:
                pred.append(temp)

        cat_gt = {}
        cat_pred = {}
        for cat in K_val:
            # Here the cat is based on the motion-averaged
            cat_gt[cat] = []
            cat_pred[cat] = []
            for p in pred:
                if int(p["mtype"]) == cat:
                    cat_pred[cat].append(p)

            for gt in d["annotations"]:
                if int(list(MOTION_TYPE.keys())[list(MOTION_TYPE.values()).index(gt["motion"]["type"])]) == cat:
                    cat_gt[cat].append(gt)

        total_num_pred += len(pred)
        total_num_rot += len(cat_pred[0])
        total_num_trans += len(cat_pred[1])

        num_gt = 0
        dtm_cat = {}
        gtm_cat = {}
        # Motion relevant parameters
        rot_type = {}
        dt_type = {}
        dt_axis = {}
        dt_origin = {}
        dt_iou = {}
        for cat in K_val:
            num_gt += len(cat_gt[cat])
            dtm_cat[cat] = -np.ones(len(cat_pred[cat]))
            gtm_cat[cat] = -np.ones(len(cat_gt[cat]))
            dt_type[cat] = -np.ones(len(cat_pred[cat]))
            rot_type[cat] = -np.ones(len(cat_pred[cat]))
            dt_axis[cat] = -np.ones(len(cat_pred[cat]))
            dt_origin[cat] = -np.ones(len(cat_pred[cat]))
            dt_iou[cat] = -np.ones(len(cat_pred[cat]))
            if len(cat_gt[cat]) == 0:
                continue
            for it in range(min(len(cat_pred[cat]), M_val[0])):
                rot_type[cat][it] = int(cat_pred[cat][it]["mtype"])
                max_iou = -1
                boxes = cat_pred[cat][it].get("bbox", None)
                for i, gt_anno in enumerate(cat_gt[cat]):
                    iou = get_iou(gt_anno["bbox"], boxes)
                    if iou > max_iou and iou > T_val[0] and gtm_cat[cat][i] == -1:
                        max_iou = iou
                        dt_iou[cat][it] = iou
                        gtm_cat[cat][i] = it
                        dtm_cat[cat][it] = i
                        # Proess the motion type
                        if int(cat_pred[cat][it]["mtype"]) == int(
                            list(MOTION_TYPE.keys())[
                                list(MOTION_TYPE.values()).index(
                                    gt_anno["motion"]["type"]
                                )
                            ]
                        ):
                            dt_type[cat][it] = 1
                        ## Process the motion axis and motion origin
                        pred_origin = np.array(cat_pred[cat][it]["morigin"])
                        gt_origin = np.array(gt_anno["motion"]["current_origin"])
                        pred_axis = np.array(cat_pred[cat][it]["maxis"])
                        gt_axis = np.array(gt_anno["motion"]["current_axis"])

                        p = pred_origin - gt_origin
                        dt_origin[cat][it] = np.linalg.norm(
                            np.cross(p, gt_axis)
                        ) / np.linalg.norm(gt_axis) / diagonal_length
                        
                        dt_axis[cat][it] = dot(gt_axis, pred_axis) / (
                            norm(gt_axis) * norm(pred_axis)
                        )
                        if dt_axis[cat][it] < 0:
                            dt_axis[cat][it] = -dt_axis[cat][it]
                        dt_axis[cat][it] = min(dt_axis[cat][it], 1.0)
                        dt_axis[cat][it] = np.arccos(dt_axis[cat][it]) / np.pi * 180
                if not dt_iou[cat][it] == -1:
                    pred_instance_eval.append((dt_iou[cat][it], dt_axis[cat][it], dt_origin[cat][it], dt_type[cat][it], rot_type[cat][it]))

    pred_instance_eval = np.array(pred_instance_eval)
    num_match = pred_instance_eval.shape[0]
    rot_mask = np.where(pred_instance_eval[:, 4]==0)[0]
    trans_mask = np.where(pred_instance_eval[:, 4]==1)[0]
    num_match_rot = rot_mask.shape[0]
    num_match_trans = trans_mask.shape[0]

    # Calculate the mean error for axis
    err_axis = round(np.mean(pred_instance_eval[:, 1]), 2)
    err_axis_r = round(np.mean(pred_instance_eval[rot_mask][:, 1]), 2)
    err_axis_t = round(np.mean(pred_instance_eval[trans_mask][:, 1]), 2)

    # Calculate the mean error for origin
    err_origin = round(np.mean(pred_instance_eval[rot_mask][:, 2]), 2)
       
    # Calculate the total error
    total_err_axis = round((err_axis * num_match + 90.0 * (total_num_pred - num_match)) / total_num_pred, 2)
    total_err_axis_r = round((err_axis_r * num_match_rot + 90.0 * (total_num_rot - num_match_rot)) / total_num_rot, 2)
    total_err_axis_t = round((err_axis_t * num_match_trans + 90.0 * (total_num_trans - num_match_trans)) / total_num_trans, 2)
    total_err_origin = round((err_origin * num_match_rot + 1.0 * (total_num_rot - num_match_rot)) / total_num_rot, 2)

    if num_match_trans == 0:
        num_match_trans = np.nan
    if total_num_trans == 0:
        total_num_trans = np.nan

    result = f"{err_axis} & {err_axis_t} & {err_axis_r} & {err_origin} & {num_match} & {num_match_trans} & {num_match_rot} & {total_err_axis} & {total_err_axis_t} & {total_err_axis_r} & {total_err_origin} & {total_num_pred} & {total_num_trans} & {total_num_rot}"

    with open(f"{cfg.OUTPUT_DIR}/ins_error_result.txt", "w") as f:
        f.write(result)

