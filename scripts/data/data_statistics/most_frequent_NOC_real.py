import glob
import json
import numpy as np
import multiprocessing
from time import time
import math

RAWDATAPATH = "/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/raw_data_real/"
TESTIDSPATH = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/raw_data_process/preprocess/testIds_real.json"
VALIDIDPATH = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/raw_data_process/preprocess/validIds_real.json"
MODELATTRPATH = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/data_statistics/real-attr.json"

# Calculates Rotation Matrix given euler angles (ZYX).
def eulerAnglesToRotationMatrix(theta):

    R_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(theta[0]), -math.sin(theta[0])],
            [0, math.sin(theta[0]), math.cos(theta[0])],
        ]
    )
    R_y = np.array(
        [
            [math.cos(theta[1]), 0, math.sin(theta[1])],
            [0, 1, 0],
            [-math.sin(theta[1]), 0, math.cos(theta[1])],
        ]
    )
    R_z = np.array(
        [
            [math.cos(theta[2]), -math.sin(theta[2]), 0],
            [math.sin(theta[2]), math.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )
    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def getAxisLineDistance(candidate_points, origin_world, axis_world):
    candidate_points = np.array(candidate_points)
    test = candidate_points - origin_world
    distance = np.linalg.norm(np.cross(test, axis_world), axis=1) / np.linalg.norm(
        axis_world
    )
    return distance


if __name__ == "__main__":
    start = time()
    # Get the test model id and valid model id
    test_ids_file = open(TESTIDSPATH)
    test_ids = json.load(test_ids_file)
    test_ids_file.close()

    valid_ids_file = open(VALIDIDPATH)
    valid_ids = json.load(valid_ids_file)
    valid_ids_file.close()

    model_paths = glob.glob(RAWDATAPATH + "*")

    model_attr_file = open(MODELATTRPATH)
    model_bound = json.load(model_attr_file)
    model_attr_file.close()

    # The data on the urdf need a coordinate transform [x, y, z] -> [z, x, y]
    modified_bound = {}
    for model in model_bound:
        modified_bound[model] = {}
        min_bound = model_bound[model]["min_bound"]
        modified_bound[model]["min_bound"] = np.array(
            [min_bound[2], min_bound[0], min_bound[1]]
        )
        max_bound = model_bound[model]["max_bound"]
        modified_bound[model]["max_bound"] = np.array(
            [max_bound[2], max_bound[0], max_bound[1]]
        )
        modified_bound[model]["scale_factor"] = (
            modified_bound[model]["max_bound"] - modified_bound[model]["min_bound"]
        )

    traintest = ["train", "valid", "test"]
    all_parts = {}

    for mode in traintest:
        if mode not in all_parts:
            all_parts[mode] = {}
        for model_path in model_paths:
            model_name = model_path.split("/")[-1]
            if mode == "train":
                if model_name in test_ids or model_name in valid_ids:
                    continue
            elif mode == "valid":
                if model_name not in valid_ids:
                    continue
            else:
                if model_name not in test_ids:
                    continue
            
            scan_json_paths = []
            scan_ids = []
            json_paths = glob.glob(f"{model_path}/origin_annotation/*")
            for json_path in json_paths:
                scan_id = json_path.split("/")[-1].split('-')[0]
                if scan_id in scan_ids:
                    continue
                scan_ids.append(scan_id)
                scan_json_paths.append(json_path)
            
            for json_path in scan_json_paths:
                # Do one statistic for each scan
                json_file = open(json_path)
                annotation = json.load(json_file)
                json_file.close()

                motions = annotation["motions"]
                # extrinsic is the transformation matrix from camera coordinate to the world coordinate
                c2w = np.reshape(annotation["camera"]["extrinsic"]["matrix"], (4, 4)).T

                for motion in motions:
                    part_label = motion["label"].strip()
                    motion_type = motion["type"]
                    origin_cam = motion["current_origin"]
                    axis_cam = motion["current_axis"]

                    origin_world = np.dot(c2w, np.array(origin_cam + [1]))[0:3]
                    axis_end_cam = list(np.array(axis_cam) + np.array(origin_cam))
                    axis_end_world = np.dot(c2w, np.array(axis_end_cam + [1]))[0:3]
                    axis_world = axis_end_world - origin_world

                    # Normalize each axis to [-0.5, 0.5]
                    origin_normalized = (
                        origin_world - modified_bound[model_name]["min_bound"]
                    ) / modified_bound[model_name]["scale_factor"] - 0.5
                    axis_normalized = axis_world / modified_bound[model_name]["scale_factor"]

                    if part_label not in all_parts[mode]:
                        all_parts[mode][part_label] = {
                            "motion_type": [],
                            "axis_world": [],
                            "origin_normalized": [],
                            "axis_normalized": [],
                        }

                    all_parts[mode][part_label]["motion_type"].append(motion_type)
                    all_parts[mode][part_label]["origin_normalized"].append(
                        origin_normalized
                    )
                    all_parts[mode][part_label]["axis_normalized"].append(axis_normalized)
                    all_parts[mode][part_label]["axis_world"].append(axis_world)

    candidate_points = np.array(
        [
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
    )

    statistics = {}
    for mode in traintest:
        if mode not in statistics:
            statistics[mode] = {}
        for part_label in all_parts[mode].keys():
            print(part_label)
            statistics[mode][part_label] = {}
            print("Processing motion type")
            # Find the most frequent motion type for the part label
            rotation_num = 0
            translation_num = 0
            for motion_type in all_parts[mode][part_label]["motion_type"]:
                if motion_type == "rotation":
                    rotation_num += 1
                else:
                    translation_num += 1
            if rotation_num > translation_num:
                statistics[mode][part_label]["motion_type"] = "rotation"
            else:
                statistics[mode][part_label]["motion_type"] = "translation"

            print("Processing motion origin")
            # Fomd the most frequent motion origin in the normalized object coordinate
            candidate_count = [0] * len(candidate_points)

            for axis_world, origin_world in zip(
                all_parts[mode][part_label]["axis_normalized"], all_parts[mode][part_label]["origin_normalized"]
            ):
                distance = getAxisLineDistance(candidate_points, origin_world, axis_world)
                candidate_count[np.argmin(distance)] += 1

            statistics[mode][part_label]["origin_normalized_number"] = candidate_count
            statistics[mode][part_label]["origin_normalized"] = list(candidate_points[np.argmax(candidate_count)])

            print("Processing motion axis")
            # Find the most frequent motion axis
            # Six base axis, then pick the one which has the smallest angle
            axis_number = [0, 0, 0]
            base_axis = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
            for axis_world in all_parts[mode][part_label]["axis_world"]:
                cos_value = np.dot(base_axis, axis_world) / np.linalg.norm(axis_world)
                cos_value = np.abs(cos_value)

                axis_number[np.argmax(cos_value)] += 1
            statistics[mode][part_label]["axis_number"] = axis_number
            statistics[mode][part_label]["axis_world"] = base_axis[
                np.argmax(axis_number)
            ].tolist()

    print(statistics)

    most_frequent_file = open(
        "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/data_statistics/most_frequent_NOP_real.json",
        "w",
    )
    json.dump(statistics, most_frequent_file)
    most_frequent_file.close()

    stop = time()
    print(str(stop - start) + " seconds")
