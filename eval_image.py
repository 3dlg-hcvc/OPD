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
NUMPREDICTION = 5
OriginThres = 0.25
AxisThres = 10.0


def getDiagLength(model_attr_path):
    model_attr_file = open(model_attr_path)
    model_bound = json.load(model_attr_file)
    model_attr_file.close()
    # The data on the urdf need a coordinate transform [x, y, z] -> [z, x, y]
    diagonal_length = {}
    for model in model_bound:
        diagonal_length[model] = model_bound[model]["diameter"]
    return diagonal_length


def img_annotation(img):
    cv_out = img.copy()
    cv_out[np.all(cv_out[:, :] == 255, axis=2)] = [0, 0, 0]
    cv_out = cv2.cvtColor(cv_out, cv2.COLOR_RGB2BGRA)
    cv_mask = cv2.cvtColor(cv_out, cv2.COLOR_BGRA2GRAY)
    _, cv_mask = cv2.threshold(cv_mask, 0, 255, cv2.THRESH_BINARY)
    cv_out[cv_mask == 0] = [0, 0, 0, 0]
    return cv_out


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
        help="path to config file",
    )
    parser.add_argument(
        "--output-dir",
        default=f"/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/analysis/image_bucket_results",
        metavar="DIR",
        help="path for training output",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="the model name",
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
        "--valid-image",
        default="/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/vis_scripts/valid_all.json",
        metavar="FILE",
        help="path to the valid image file",
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
    os.makedirs(f"{cfg.OUTPUT_DIR}/{args.model_name}", exist_ok=True)

    # metadata is just used for the category infomation
    dataset_key = cfg.DATASETS.TEST[0]
    motion_metadata = MetadataCatalog.get(dataset_key)

    # predictor = DefaultPredictor(cfg)
    dataset_dicts = DatasetCatalog.get(dataset_key)

    from fvcore.common.file_io import PathManager

    with PathManager.open(cfg.INFERENCE_FILE, "rb") as f:
        buffer = io.BytesIO(f.read())
    predictions = torch.load(buffer)

    pred_gt_match = []

    # Draw the valid image
    T_val = [0.5]
    R_val = np.linspace(
        0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
    )
    K_val = [0, 1, 2]
    A_val = [1e5 ** 2]
    M_val = [100]
    T = len(T_val)
    R = len(R_val)
    K = len(K_val)
    A = len(A_val)
    M = len(M_val)

    valid_image_file = open(args.valid_image)
    selection = json.load(valid_image_file)
    valid_image_file.close()

    diagonal_lengths = getDiagLength(args.model_attr_path)

    count = 0
    for d in dataset_dicts:
        if os.path.basename(d["file_name"]).split(".")[0] not in selection:
            continue
        img_path = d["file_name"]
        model_name = (os.path.basename(img_path)).split("-")[0]
        diagonal_length = diagonal_lengths[model_name]

        try:
            prediction = predictions[d["image_id"] - 1]
        except Exception as e:
            raise d["image_id"] - 1

        # im is loaded by cv2 in BGR, Visualizer needs RGB format
        instance_number = len(prediction["instances"])
        M_val[0] = min(instance_number, 100)
        ap_detect = -np.ones((K, R))
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
            cat_gt[cat] = []
            cat_pred[cat] = []
            for p in pred:
                if int(p["category_id"]) == cat:
                    cat_pred[cat].append(p)

            for gt in d["annotations"]:
                if int(gt["category_id"]) == cat:
                    cat_gt[cat].append(gt)

        num_gt = 0
        dtm_cat = {}
        gtm_cat = {}
        # Motion relevant parameters
        rot_type = {}
        dt_type = {}
        dt_axis = {}
        dt_origin = {}
        for cat in K_val:
            num_gt += len(cat_gt[cat])
            dtm_cat[cat] = -np.ones(len(cat_pred[cat]))
            gtm_cat[cat] = -np.ones(len(cat_gt[cat]))
            dt_type[cat] = -np.ones(len(cat_pred[cat]))
            rot_type[cat] = -np.ones(len(cat_pred[cat]))
            dt_axis[cat] = -np.ones(len(cat_pred[cat]))
            dt_origin[cat] = -np.ones(len(cat_pred[cat]))
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
                        # For the origins of the translation joint, all are valid
                        if dt_origin[cat][it] <= OriginThres or rot_type[cat][it] == 1:
                            dt_origin[cat][it] = 1
                        else:
                            dt_origin[cat][it] = -1
                        
                        dt_axis[cat][it] = dot(gt_axis, pred_axis) / (
                            norm(gt_axis) * norm(pred_axis)
                        )
                        if dt_axis[cat][it] < 0:
                            dt_axis[cat][it] = -dt_axis[cat][it]
                        dt_axis[cat][it] = min(dt_axis[cat][it], 1.0)
                        dt_axis[cat][it] = np.arccos(dt_axis[cat][it]) / np.pi * 180
                        if dt_axis[cat][it] <= AxisThres:
                            dt_axis[cat][it] = 1
                        else:
                            dt_axis[cat][it] = -1


            tps = dtm_cat[cat] != -1
            tp_sum = np.cumsum(tps, axis=0).astype(dtype=float)
            fps = dtm_cat[cat] == -1
            fp_sum = np.cumsum(fps, axis=0).astype(dtype=float)
            rc = tp_sum / len(cat_gt[cat])
            pr = tp_sum / (tp_sum + fp_sum)
            q = np.zeros(R)

            for i in range(len(tp_sum) - 1, 0, -1):
                if pr[i] > pr[i - 1]:
                    pr[i - 1] = pr[i]

            inds = np.searchsorted(rc, R_val, side="left")

            try:
                for ri, pi in enumerate(inds):
                    q[ri] = pr[pi]
            except:
                pass

            ap_detect[cat] = q

        if not ap_detect[ap_detect > -1].shape[0] == 0:
            map_detect = np.mean(ap_detect[ap_detect > -1])
        else:
            map_detect = -1
        output_instance = d["file_name"].split("/")[-1]
        pred_gt_match.append((output_instance, map_detect))

    pred_gt_match.sort(key=lambda x: -x[1])

    scores = [x[1] for x in pred_gt_match]

    # Divide the instances into 5 bucket and store them into a json file
    buckets = {"80-100": [], "60-80": [], "40-60": [], "20-40": [], "0-20": []}
    for (image_name, score) in pred_gt_match:
        if score > 0.8 and score <= 1.0:
            buckets["80-100"].append(image_name)
        elif score > 0.6 and score <= 0.8:
            buckets["60-80"].append(image_name)
        elif score > 0.4 and score <= 0.6:
            buckets["40-60"].append(image_name)
        elif score > 0.2 and score <= 0.4:
            buckets["20-40"].append(image_name)
        elif score >= 0.0 and score <= 0.2:
            buckets["0-20"].append(image_name)

    print("Below is the distribution of the number of image in the buckets:")
    print("80-100 60-80 40-60 20-40 0-20")
    print(f'{len(buckets["80-100"])} {len(buckets["60-80"])} {len(buckets["40-60"])} {len(buckets["20-40"])} {len(buckets["0-20"])}')

    for range in buckets.keys():
        with open(
            f"{cfg.OUTPUT_DIR}/{args.model_name}/eval_image_{args.model_name}_{range}.json",
            "w+",
        ) as fp:
            json.dump(buckets[range], fp)
