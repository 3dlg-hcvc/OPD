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
        "--valid-image",
        required=True,
        metavar="FILE",
        help="path to the valid image file",
    )
    parser.add_argument(
        "--update-all",
        required=False,
        default=False,
        action="store_true",
        help="if True, update both annotation image and the files, or only update the annotation",
    )
    parser.add_argument(
        "--score-threshold",
        required=True,
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
        "--is-real",
        action="store_true",
        help="indicating whether to visualize the gt for the MotionREAL dataset",
    )
    parser.add_argument(
        "--no_mask",
        action="store_true",
        help="indicating whether to visualize the mask of the prediction",
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

    valid_image_file = open(args.valid_image)
    selection = json.load(valid_image_file)
    valid_image_file.close()

    diagonal_lengths = getDiagLength(args.model_attr_path)

    count = 0
    for d in dataset_dicts:
        if args.is_real:
            # intrinsic_matrix = np.reshape(d["camera"]["intrinsic"]["matrix"], (3, 3), order='F')
            intrinsic_matrix = np.array([[283.18526475694443, 0., 126.65098741319443], [0., 283.18526475694443, 128.45118272569442],[ 0., 0., 1.]]) 
            line_length = 0.2
        else:
            intrinsic_matrix = None
            line_length = 1

        if os.path.basename(d["file_name"]).split(".")[0] not in selection:
            continue
        img_path = d["file_name"]
        model_name = (os.path.basename(img_path)).split("-")[0]
        diagonal_length = diagonal_lengths[model_name]
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
        else:
            alpha = img[:, :, 3]
            img[alpha != 255, :] = [255, 255, 255, 0]
            img = img[:, :, :3]
            cv_in = img.copy()
            cv_in = cv2.cvtColor(cv_in, cv2.COLOR_BGR2RGB)

        if args.transparent:
            background = np.zeros_like(cv_in).astype(np.uint8)
            cv_in = background.copy()

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

        map_detect = np.mean(ap_detect[ap_detect > -1])

        # Calculate the overall detect precision (micro average)
        sum_tps = 0
        sum_fps = 0
        for cat in K_val:
            if cat not in dtm_cat:
                continue
            det_tps = dtm_cat[cat] != -1
            sum_tps = sum_tps + np.sum(det_tps).astype(dtype=float)
            det_fps = dtm_cat[cat] == -1
            sum_fps = sum_fps + np.sum(det_fps).astype(dtype=float)

        if sum_fps == 0 and sum_tps == 0:
            det_pr = -1
        else:
            det_pr = sum_tps / (sum_tps + sum_fps)

        # Caulculate the overall recall and f1 score for the detection
        if num_gt == 0:
            det_rc = -1
            det_f1 = -1
        else:
            det_rc = sum_tps / num_gt
            if det_pr == -1 or det_pr + det_rc == 0:
                det_f1 = -1
            else:
                det_f1 = (2 * det_pr * det_rc) / (det_pr + det_rc)

        # Calculate the overall motion precision (micro average)
        sum_tps = 0
        sum_fps = 0
        for cat in K_val:
            if cat not in dtm_cat:
                continue
            motion_tps = np.logical_and.reduce((dtm_cat[cat] != -1, dt_type[cat] != -1, dt_origin[cat] != -1, dt_axis[cat] != -1))
            sum_tps = sum_tps + np.sum(motion_tps).astype(dtype=float)
            motion_fps = np.logical_not(np.logical_and.reduce((dtm_cat[cat] != -1, dt_type[cat] != -1, dt_origin[cat] != -1, dt_axis[cat] != -1)))
            sum_fps = sum_fps + np.sum(motion_fps).astype(dtype=float)

        if sum_fps == 0 and sum_tps == 0:
            motion_pr = -1
        else:
            motion_pr = sum_tps / (sum_tps + sum_fps)

        # # Caulculate the overall recall and f1 score for the motion
        if num_gt == 0:
            motion_rc = -1
            motion_f1 = -1
        else:
            motion_rc = sum_tps / num_gt
            if motion_pr == -1 or motion_pr + motion_rc == 0:
                motion_f1 = -1
            else:
                motion_f1 = (2 * motion_pr * motion_rc) / (motion_pr + motion_rc)
        
        
        # Calculate the overall type precision (micro average)
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

        i = 0
        visited = []
        for cat in K_val:
            if cat not in dtm_cat:
                continue

            if i >= NUMPREDICTION:
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
                        pred_gt_match[output_instance], is_real=args.is_real, intrinsic_matrix=intrinsic_matrix, line_length=line_length, no_mask=args.no_mask, diagonal_length=diagonal_length
                    )
                    visited.append(cat_pred[cat][pred_idx])

                    if v:
                        pred_gt_match[output_instance]["map_detect"] = map_detect
                        # Results for detection
                        pred_gt_match[output_instance]["detect_precision"] = det_pr
                        pred_gt_match[output_instance]["detect_recall"] = det_rc
                        pred_gt_match[output_instance]["detect_f1"] = det_f1
                        # Results for all motion params & det
                        pred_gt_match[output_instance]["motion_precision"] = motion_pr
                        pred_gt_match[output_instance]["motion_recall"] = motion_rc
                        pred_gt_match[output_instance]["motion_f1"] = motion_f1
                        # Results for motion type
                        pred_gt_match[output_instance]["type_precision"] = type_pr
                        pred_gt_match[output_instance]["total_pred"] = len(pred)
                        if args.update_all:
                            if args.transparent:
                                cv_out = img_annotation(v.get_image())
                            else:
                                cv_out = v.get_image()
                                cv_out = cv2.cvtColor(cv_out, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(f"{cfg.OUTPUT_DIR}/{output_instance}", cv_out)
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
            if i >= NUMPREDICTION or j >= NUMPREDICTION:
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
            v = v.draw_pred_instance(pred[j], d, pred_gt_match[output_instance], is_real=args.is_real, intrinsic_matrix=intrinsic_matrix, line_length=line_length, no_mask=args.no_mask, diagonal_length=diagonal_length)
            if v:
                pred_gt_match[output_instance]["map_detect"] = -1
                # Results for detection
                pred_gt_match[output_instance]["detect_precision"] = det_pr
                pred_gt_match[output_instance]["detect_recall"] = det_rc
                pred_gt_match[output_instance]["detect_f1"] = det_f1
                # Results for all motion params & det
                pred_gt_match[output_instance]["motion_precision"] = motion_pr
                pred_gt_match[output_instance]["motion_recall"] = motion_rc
                pred_gt_match[output_instance]["motion_f1"] = motion_f1
                # Results for motion type
                pred_gt_match[output_instance]["type_precision"] = type_pr
                pred_gt_match[output_instance]["total_pred"] = len(pred)
                if args.update_all:
                    if args.transparent:
                        cv_out = img_annotation(v.get_image())
                    else:
                        cv_out = v.get_image()
                        cv_out = cv2.cvtColor(cv_out, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f"{cfg.OUTPUT_DIR}/{output_instance}", cv_out)
                i = i + 1
            else:
                pred_gt_match.pop(output_instance)
            j = j + 1
        count += 1
        print(f"Save {count}/{len(selection)} images to {cfg.OUTPUT_DIR}")
    print('Saving annotations')
    with open(
        f"{cfg.OUTPUT_DIR}/../annotations/instance_render_{os.path.basename(cfg.OUTPUT_DIR)}.json",
        "w+",
    ) as fp:
        json.dump(pred_gt_match, fp)
    print('Done')
