import cv2
import argparse
import os
import datetime
import numpy as np
import json
import yaml

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger

from motlib import register_motion_instances, MotionVisualizer, add_motionnet_config

def get_parser():
    parser = argparse.ArgumentParser(description="Train motion net")
    parser.add_argument(
        "--json-file",
        default="/local-scratch/localhome/yma50/Development/motionnet/2DMotion/prior_pred.json",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--data-path",
        required=False,
        default="/cs/3dlg-project/3dlg-hcvc/motionnet/Dataset/drawer",
        help="path containing motion datasets",
    )
    parser.add_argument(
        "--output-dir",
        default=f"prior",
        metavar="DIR",
        help="path for training output",
    )
    parser.add_argument(
        "--transparent",
        required=False,
        default=False,
        action="store_true",
        help="input binary inference file",
    )
    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.json_file, "r") as f:
         predictions = json.load(f)

    count = 0
    for key, all_joints in predictions.items():
        for part_id, joint in all_joints.items():
            img_id, pose_id, view_id = key.split('_')
            basepath = os.path.join(args.data_path, img_id, pose_id)
            img_path = os.path.join(basepath, "rgb", view_id+".png")
            yaml_path = os.path.join(basepath, "gt.yml")

            img = cv2.imread(img_path)

            black_pixels = np.where(
                (img[:, :, 0] == 0) & 
                (img[:, :, 1] == 0) & 
                (img[:, :, 2] == 0)
            )

            img[black_pixels] = [255, 255, 255]
            img = img[:, :, :3]

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = cv2.resize(img, (256, 256))

            with open(yaml_path, "r") as f:
                meta_instance = yaml.load(f, Loader=yaml.Loader)

            projection = np.asarray(meta_instance["frame_"+view_id]['projMat']).reshape((4,4)).transpose()

            joint_pt = np.asarray(joint["start_pt"])
            joint_axis = np.asarray(joint["axis"])

            joint_axis = joint_axis / np.linalg.norm(joint_axis)

            start = np.asarray(joint["start_pt"])
            end = np.asarray(joint["end_pt"])

            visualizer = MotionVisualizer(
                img, metadata=None, scale=2
            )

            anno_dict = {}
            anno_dict["origin"] = start
            anno_dict["start"] = start
            anno_dict["end"] = end

            anno_dict["start_2d"] = np.asarray(joint["2d_start"])/2
            anno_dict["end_2d"] = np.asarray(joint["2d_end"])/2

            anno_dict["axises"] = joint_axis

            anno_dict["error"] = np.asarray(joint["error"])

            anno_dict["projMat"] = projection
            anno_dict["img_size"] = img.shape[0]

            vis = visualizer.draw_prior(anno_dict)
            cv_out = vis.get_image()
            cv_out = cv2.cvtColor(cv_out, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'{args.output_dir}/{key}__{part_id}.png', cv_out)
            count += 1
            print(key)
