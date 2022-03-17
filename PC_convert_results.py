import os
import torch
import argparse
from fvcore.common.file_io import PathManager
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetCatalog
from detectron2.config import get_cfg
from time import time
import math
import io
import h5py
import numpy as np
from alive_progress import alive_bar


from motlib import add_motionnet_config, register_motion_instances

# TEST_INFERENCE_FILE_PATH = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/train_output/instances_predictions.pth"
# TEST_INFERENCE_FILE_PATH = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/ancsh_result.pth"
# We only care about the evaluation on the bbx metric
dummy_segmentation = {'size': [256, 256], 'counts': 'g]W1;a76E<H6G:D<K4K6M3N1O2N2O1O010O1O011OO02O0O10000O100O01000O002OO01O01O0O1O2N1O1O2N2N1O2N1O1O2N1O2N1N2N2M4M3L3M4L3M3N4J4N3L3M3N4K3M4KXQ8'}


def get_parser():
    parser = argparse.ArgumentParser(description="Train motion net")
    parser.add_argument(
        "--result-path",
        default="/localhome/hja40/Desktop/Research/proj-motionnet/PC_motion_prediction/pc_output/final_result.h5",
        metavar="DIR",
        help="path containing motion datasets",
    )
    parser.add_argument(
        "--data-path",
        default="/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/MotionDataset_h5_6.11",
        metavar="DIR",
        help="path containing motion datasets",
    )
    parser.add_argument(
        "--test",
        default=False,
        action="store_true",
        help="If True, this is the result on the test set",
    )
    parser.add_argument(
        "--max_K",
        default=5,
        type=int,
        help="indicatet the max number for the segmentation",
    )
    return parser

def register_datasets(data_path, Name):
    dataset_keys = [Name]
    for dataset_key in dataset_keys:
        json = f"{data_path}/annotations/{dataset_key}.json"
        imgs = f"{data_path}/{dataset_key.split('_')[-1]}"
        register_motion_instances(dataset_key, {}, json, imgs)

def create_image_mapper(dataset_dicts):
    image_list = []
    index = 0
    for d in dataset_dicts:
        image_name = os.path.basename(d["file_name"]).split('.')[0]
        image_id = d["image_id"]
        image_list.append({"image_name": image_name, "image_id": image_id})
        index += 1
        assert image_id == index
    
    return image_list

if __name__ == "__main__":
    start = time()

    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    if args.test:
        dataset_key = "MotionNet_test"
    else:
        dataset_key = "MotionNet_valid"
    register_datasets(args.data_path, dataset_key)
    dataset_dicts = DatasetCatalog.get(dataset_key)

    image_list = create_image_mapper(dataset_dicts)

    # with PathManager.open(TEST_INFERENCE_FILE_PATH, "rb") as f:
    #     buffer = io.BytesIO(f.read())
    # inference_file = torch.load(buffer)

    # cat_ids = []
    # for i in inference_file:
    #     for j in i["instances"]:
    #         cat_ids.append(j["mtype"])

    # import pdb
    # pdb.set_trace()

    pc_results = h5py.File(args.result_path, "r")
    pc_images = pc_results.keys()


    predictions = []
    num_invalid = 0
    with alive_bar(len(image_list)) as bar:
        for image in image_list:
            image_name = image["image_name"]
            image_id = image["image_id"]

            if "bg" in image_name:
                image_name = image_name.split("+")[0]

            if image_name not in pc_images:
                print(f"Image not in the PC result {image_name}")
                num_invalid += 1
                predictions.append({"image_id": image_id, "instances": []})
            else:
                instances = []
                result = pc_results[image_name]
                is_valid = result["is_valid"][:]
                cat_map = result["cat_map"][:]
                mtype_map = result["mtype_map"][:]
                maxis_map = result["maxis_map"][:]
                morigin_map = result["morigin_map"][:]
                bbx_map = result["bbx_map"][:]
                for index in range(args.max_K):
                    if is_valid[index] == False:
                        continue
                    instance = {}
                    instance["image_id"] = image_id
                    instance["score"] = 1.0
                    instance["category_id"] = int(cat_map[index])
                    instance["bbox"] = list(bbx_map[index])
                    instance["segmentation"] = dummy_segmentation
                    instance["mtype"] = int(mtype_map[index])
                    instance["maxis"] = list(maxis_map[index])
                    instance["morigin"] = list(morigin_map[index])
                    instances.append(instance)
                predictions.append({"image_id": image_id, "instances": instances})
            bar()
            
    with PathManager.open("pc_result.pth", "wb") as f:
        torch.save(predictions, f)

    stop = time()
    print(f"Total Invalid instances number: {num_invalid}")
    print(f'Total time duration: {stop - start}')