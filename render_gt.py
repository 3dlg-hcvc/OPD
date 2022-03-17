import cv2
import argparse
import os
import datetime
import numpy as np
import json

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger

from motlib import register_motion_instances, MotionVisualizer, add_motionnet_config

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_motionnet_config(cfg)
    cfg.merge_from_file(args.config_file)

    # Output directory
    cfg.OUTPUT_DIR = args.output_dir

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
        "--data-path",
        required=True,
        metavar="DIR",
        help="path containing motion datasets",
    )
    parser.add_argument(
        "--output-dir",
        default=f"train_output/{datetime.datetime.now().isoformat()}",
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
    parser.add_argument(
        "--valid-image",
        required=True,
        metavar="FILE",
        help="path to the valid image file",
    )
    parser.add_argument(
        "--is-real",
        action="store_true",
        help="indicating whether to visualize the gt for the MotionREAL dataset",
    )
    return parser

def register_datasets(data_path, cfg):
    dataset_keys = cfg.DATASETS.TRAIN + cfg.DATASETS.TEST
    for dataset_key in dataset_keys:
        json = f"{data_path}/annotations/{dataset_key}.json"
        imgs = f"{data_path}/{dataset_key.split('_')[-1]}"
        register_motion_instances(dataset_key, {}, json, imgs)

if __name__ == '__main__':
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    register_datasets(args.data_path, cfg)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{cfg.OUTPUT_DIR}/../annotations", exist_ok=True)

    # Register the valid dataset
    print(cfg.DATASETS.TEST)
    dataset_key = cfg.DATASETS.TEST[0]
    valid_json_file = f"{args.data_path}/annotations/{dataset_key}.json"
    valid_image_root = f"{args.data_path}/{dataset_key.split('_')[-1]}"
    
    # metadata is just used for the category infomation
    motion_metadata = MetadataCatalog.get(dataset_key)
    dataset_dicts = DatasetCatalog.get(dataset_key)

    part_ids = {}
    valid_image_file = open(args.valid_image)
    selection = json.load(valid_image_file)
    valid_image_file.close()
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
        img = cv2.imread(d['file_name'], cv2.IMREAD_UNCHANGED)
        if args.transparent:
            background = np.zeros_like(img).astype(np.uint8)
            cv_in = background.copy()
        else:
            alpha = img[:, :, 3]
            img[alpha!=255, :] = [255, 255, 255, 0]
            img = img[:, :, :3]
            cv_in = img.copy()
            cv_in = cv2.cvtColor(cv_in, cv2.COLOR_BGR2RGB)
        instance_number = len(d['annotations'])
        for i in range(instance_number):
            visualizer = MotionVisualizer(
                cv_in, metadata=motion_metadata, scale=2
            )
            instance_name = f'{(d["file_name"].split("/")[-1]).split(".")[0]}__{i}.png'
            part_ids[instance_name]={}
            vis = visualizer.draw_gt_instance(d['annotations'][i], part_ids[instance_name], is_real=args.is_real, intrinsic_matrix=intrinsic_matrix, line_length=line_length)
            cv_out = vis.get_image()
            if args.transparent:
                cv_out[np.all(cv_out[:, :] == 255, axis=2)] = [0, 0, 0]
                cv_out = cv2.cvtColor(cv_out, cv2.COLOR_RGB2BGRA)
                cv_mask = cv2.cvtColor(cv_out, cv2.COLOR_BGRA2GRAY)
                _, cv_mask = cv2.threshold(cv_mask, 0, 255, cv2.THRESH_BINARY)
                cv_out[cv_mask == 0] = [0, 0, 0, 0]
            else:
                cv_out = cv2.cvtColor(cv_out, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'{cfg.OUTPUT_DIR}/{instance_name}', cv_out)
        count += 1
        print(f'Save {count}/{len(selection)} images to {cfg.OUTPUT_DIR}')
    
    with open(f'{cfg.OUTPUT_DIR}/../annotations/instance_render_{os.path.basename(cfg.OUTPUT_DIR)}.json', 'w+') as fp:
        json.dump(part_ids, fp, indent=4)
