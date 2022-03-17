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

from motlib import add_motionnet_config, register_motion_instances

# TEST_INFERENCE_FILE_PATH = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/train_output/instances_predictions.pth"
# TEST_INFERENCE_FILE_PATH = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/ancsh_result.pth"
# We only care about the evaluation on the bbx metric
dummy_segmentation = {'size': [256, 256], 'counts': 'g]W1;a76E<H6G:D<K4K6M3N1O2N2O1O010O1O011OO02O0O10000O100O01000O002OO01O01O0O1O2N1O1O2N2N1O2N1O1O2N1O2N1N2N2M4M3L3M4L3M3N4J4N3L3M3N4K3M4KXQ8'}

# drawer: 0, door 1, lid: 2
# Only one moving part is 1 in this case
PART_CAT_MAP = [1]
# rotation: 0, translation: 1
MOTION_TYPE_MAP = [0]

def getFocalLength(FOV, height, width=None):
    # FOV is in radius, should be vertical angle
    if width == None:
        f = height / (2 * math.tan(FOV / 2))
        return f
    else:
        fx = height / (2 * math.tan(FOV / 2))
        fy = fx / height * width
        return (fx, fy)


def get_parser():
    parser = argparse.ArgumentParser(description="Train motion net")
    parser.add_argument(
        "--ancsh-result-path",
        default="/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/ANCSH-pytorch/data/evaluation/motionnet_val/prediction.h5",
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

    FOV = 50
    img_width = 256
    img_height = 256
    fx, fy = getFocalLength(FOV / 180 * math.pi, img_height, img_width)
    cy = img_height / 2
    cx = img_width / 2

    ancsh_results = h5py.File(args.ancsh_result_path, "r")
    ancsh_images = ancsh_results.keys()

    # val_models = []
    # for image in ancsh_images:
    #     model_name = image.split('_')[1]
    #     if model_name not in val_models:
    #         val_models.append(model_name)

    predictions = []
    for image in image_list:
        image_name = image["image_name"]
        image_id = image["image_id"]

        if "bg" in image_name:
            image_name = image_name.split("+")[0]

        # # Only for special case for ancsh not complete val
        # splits = image_name.split('-')
        # if len(splits) == 4:
        #     image_name = f"OneDoor_{splits[0]}_{splits[2]}_{splits[3]}"
        # else:
        #     image_name = f"OneDoor_{splits[0]}_{splits[1]}_{splits[2]}"

        if image_name not in ancsh_images:
            print(f"Image not in the ANCSH result {image_name}")
            predictions.append({"image_id": image_id, "instances": []})
        else:
            instances = []
            result = ancsh_results[image_name]
            if result["is_valid"][0] == False or result["joint_is_valid"] == False:
                predictions.append({"image_id": image_id, "instances": []})
            else:
                # Project the 3dbbx back to image-based
                num_moving_part = len(result["pred_joint_axis_cam"])
                for part_index in range(num_moving_part):
                    instance = {}
                    instance["image_id"] = image_id
                    instance["category_id"] = PART_CAT_MAP[part_index]
                    instance["score"] = 1.0

                    # Get the bbx
                    # for the bounding box, ignore the base part
                    ## Use the bounding box from npcs
                    # bbx_cam = result["pred_cam_3dbbx"][part_index+1]
                    ## Use the bounding box directly from input camera PC
                    pred_seg_per_point = result["pred_seg_per_point"][:]
                    point_index = np.where(pred_seg_per_point == part_index+1)[0]
                    bbx_cam = result["camcs_per_point"][point_index]
                    # bbx_cam  
                    bbx_cam[:, 0] = bbx_cam[:, 0] * fx / (-bbx_cam[:, 2]) + cx
                    bbx_cam[:, 1] = -(bbx_cam[:, 1] * fy / (-bbx_cam[:, 2])) + cy
                    x_min = np.float64(np.min(bbx_cam[:, 0]))
                    x_max = np.float64(np.max(bbx_cam[:, 0]))
                    y_min = np.float64(np.min(bbx_cam[:, 1]))
                    y_max = np.float64(np.max(bbx_cam[:, 1]))
                    instance["bbox"] = [x_min, y_min, x_max - x_min, y_max - y_min]
                    # import pdb
                    # pdb.set_trace()
                    # Set the dummy segmentation 
                    instance["segmentation"] = dummy_segmentation

                    # Get the segmentation info (it can be anything), we don't care the evaluation metric on segmentation
                    # Because the camera PC is sampled and cannot directly convert back to the image
                    instance["mtype"] = MOTION_TYPE_MAP[part_index]
                    instance["maxis"] = list(result["pred_joint_axis_cam"][part_index])
                    instance["morigin"] = list(result["pred_joint_pt_cam"][part_index])
                    instances.append(instance)
                predictions.append({"image_id": image_id, "instances": instances})
        
    with PathManager.open("ancsh_result.pth", "wb") as f:
        torch.save(predictions, f)

    stop = time()

    print(f'Total time duration: {stop - start}')