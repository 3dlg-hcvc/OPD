from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.utils.logger import setup_logger
import pycocotools.mask as mask_util

import os
import io
import cv2
import torch
import argparse
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt

from motlib import (
    MotionTrainer,
    add_motionnet_config,
    register_motion_instances,
    MotionVisualizer,
)

MOTION_TYPE = {0: "rotation", 1: "translation"}


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
        "--inference-file",
        required=True,
        metavar="FILE",
        help="input binary inference file",
    )
    parser.add_argument(
        "--transparent",
        required=False,
        default=False,
        action="store_true",
        help="input binary inference file",
    )
    parser.add_argument(
        "--depth",
        required=False,
        default=False,
        action="store_true",
        help="output to depth images",
    )
    parser.add_argument(
        "--mask",
        required=False,
        default=False,
        action="store_true",
        help="indicate whether to render mask,",
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
    os.makedirs(f"{cfg.OUTPUT_DIR}/../annotations", exist_ok=True)

    if args.mask:
        os.makedirs(f"{cfg.OUTPUT_DIR}/../masks", exist_ok=True)
        os.makedirs(f"{cfg.OUTPUT_DIR}/../rgb", exist_ok=True)
        os.makedirs(f"{cfg.OUTPUT_DIR}/../depth", exist_ok=True)

    # metadata is just used for the category infomation
    dataset_key = cfg.DATASETS.TEST[0]
    motion_metadata = MetadataCatalog.get(dataset_key)

    # predictor = DefaultPredictor(cfg)
    dataset_dicts = DatasetCatalog.get(dataset_key)

    from fvcore.common.file_io import PathManager

    with PathManager.open(cfg.INFERENCE_FILE, "rb") as f:
        buffer = io.BytesIO(f.read())
    predictions = torch.load(buffer)

    pred_gt_match = {}

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

    selection = [
        "46334-0-3-4+bg2",
    ]
    count = 0
    for d in dataset_dicts:
        if os.path.basename(d["file_name"]).split(".")[0] not in selection:
            continue

        if args.mask:
            img_path = d["file_name"]
            print(img_path)
            depth_path = os.path.join(
                os.path.dirname(os.path.dirname(img_path)),
                "depth",
                os.path.splitext(os.path.basename(img_path))[0] + "_d.png",
            )
            os.system(f'cp {d["file_name"]} {cfg.OUTPUT_DIR}/../rgb/')
            os.system(f"cp {depth_path} {cfg.OUTPUT_DIR}/../depth/")

        img_path = d["file_name"]
        if args.depth:
            img_path = os.path.join(
                os.path.dirname(os.path.dirname(img_path)),
                "depth",
                os.path.splitext(os.path.basename(img_path))[0] + "_d.png",
            )
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if args.depth:
            clamp = 5000.0
            cm = plt.get_cmap("gray")
            gray = cm(img.astype("float") / clamp)
            gray *= 255
            gray = gray.astype("uint8")
            gray = cv2.cvtColor(gray, cv2.COLOR_RGBA2BGR)
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
            cv_in = gray.copy()
        elif args.transparent:
            background = np.zeros_like(img).astype(np.uint8)
            cv_in = background.copy()
        else:
            alpha = img[:, :, 3]
            img[alpha != 255, :] = [255, 255, 255, 0]
            img = img[:, :, :3]
            cv_in = img.copy()
            cv_in = cv2.cvtColor(cv_in, cv2.COLOR_BGR2RGB)

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

        type_matched = 0
        dtm_cat = {}
        gtm_cat = {}
        dt_type = {}
        for cat in K_val:
            if len(cat_gt[cat]) == 0:
                continue
            dtm_cat[cat] = -np.ones(len(cat_pred[cat]))
            gtm_cat[cat] = -np.ones(len(cat_gt[cat]))
            dt_type[cat] = -np.ones(len(cat_pred[cat]))
            for it in range(min(len(cat_pred[cat]), M_val[0])):
                max_iou = -1
                boxes = cat_pred[cat][it].get("bbox", None)
                for i, gt_anno in enumerate(cat_gt[cat]):
                    iou = get_iou(gt_anno["bbox"], boxes)
                    if iou > max_iou and iou > T_val[0] and gtm_cat[cat][i] == -1:
                        max_iou = iou
                        gtm_cat[cat][i] = it
                        dtm_cat[cat][it] = i
                        if int(cat_pred[cat][it]["mtype"]) == int(
                            list(MOTION_TYPE.keys())[
                                list(MOTION_TYPE.values()).index(
                                    gt_anno["motion"]["type"]
                                )
                            ]
                        ):
                            dt_type[cat][it] = 0

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

        sum_tps = 0
        sum_fps = 0
        for cat in K_val:
            if cat not in dt_type:
                continue
            type_tps = dt_type[cat] != -1
            sum_tps = sum_tps + np.sum(type_tps).astype(dtype=float)
            type_fps = dt_type[cat] == -1
            sum_fps = sum_fps + np.sum(type_fps).astype(dtype=float)

        if sum_fps == 0 and sum_tps == 0:
            type_pr = 0
        else:
            type_pr = sum_tps / (sum_tps + sum_fps)

        map_detect = np.mean(ap_detect[ap_detect > -1])

        i = 0
        visited = []
        for cat in K_val:
            if cat not in dtm_cat:
                continue

            if i > 5:
                break
            for j in range(len(dtm_cat[cat])):
                gt_idx = int(dtm_cat[cat][j])
                if gt_idx > -1:
                    pred_idx = int(gtm_cat[cat][gt_idx])
                    if pred_idx == -1:
                        print("Error pred_idx")
                    v = MotionVisualizer(
                        cv_in,
                        metadata=motion_metadata,
                        scale=2,
                        instance_mode=ColorMode.IMAGE,  # remove the colors of unsegmented pixels
                    )
                    output_instance = (
                        f'{(d["file_name"].split("/")[-1]).split(".")[0]}__{i}.png'
                    )
                    pred_gt_match[output_instance] = {}
                    v = v.draw_pred_instance(
                        cat_pred[cat][pred_idx],
                        cat_gt[cat][gt_idx],
                        pred_gt_match[output_instance],
                    )
                    visited.append(cat_pred[cat][pred_idx])

                    if v:
                        pred_gt_match[output_instance]["map_detect"] = map_detect
                        pred_gt_match[output_instance]["type_precision"] = type_pr
                        if args.transparent:
                            cv_out = img_annotation(v.get_image())
                        else:
                            cv_out = v.get_image()
                            cv_out = cv2.cvtColor(cv_out, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(f"{cfg.OUTPUT_DIR}/{output_instance}", cv_out)

                        # Draw mask
                        if args.mask:
                            mask = mask_util.decode(
                                cat_pred[cat][pred_idx].get("segmentation")
                            )
                            cv2.imwrite(
                                f"{cfg.OUTPUT_DIR}/../masks/{output_instance}", mask
                            )

                        i = i + 1
                    else:
                        pred_gt_match.pop(output_instance)
        j = 0
        while True:
            if len(pred) <= j:
                break
            if pred[j] in visited:
                j = j + 1
                continue
            if i > 5 or j > 5:
                break
            v = MotionVisualizer(
                cv_in,
                metadata=motion_metadata,
                scale=2,
                instance_mode=ColorMode.IMAGE,  # remove the colors of unsegmented pixels
            )
            output_instance = (
                f'{(d["file_name"].split("/")[-1]).split(".")[0]}__{i}.png'
            )
            pred_gt_match[output_instance] = {}
            v = v.draw_pred_instance(pred[j], d, pred_gt_match[output_instance])
            if v:
                pred_gt_match[output_instance]["map_detect"] = -1
                pred_gt_match[output_instance]["type_precision"] = type_pr
                if args.transparent:
                    cv_out = img_annotation(v.get_image())
                else:
                    cv_out = v.get_image()
                    cv_out = cv2.cvtColor(cv_out, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"{cfg.OUTPUT_DIR}/{output_instance}", cv_out)

                # Draw mask
                if args.mask:
                    mask = mask_util.decode(pred[j].get("segmentation"))
                    cv2.imwrite(f"{cfg.OUTPUT_DIR}/../masks/{output_instance}", mask)

                i = i + 1
            else:
                pred_gt_match.pop(output_instance)
            j = j + 1
        count += 1
        print(f"Save {count}/{len(dataset_dicts)} images to {cfg.OUTPUT_DIR}")
    with open(
        f"{cfg.OUTPUT_DIR}/../annotations/instance_render_{os.path.basename(cfg.OUTPUT_DIR)}.json",
        "w+",
    ) as fp:
        json.dump(pred_gt_match, fp)
