import glob
import json
import numpy as np
import multiprocessing
from time import time
import math

RAWDATAPATH = "/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/raw_data_6.1/"
TESTIDSPATH = '/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/raw_data_process/preprocess/testIds.json'
VALIDIDPATH = '/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/raw_data_process/preprocess/validIds.json'

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


def getALLPoints(min_bound, max_bound, c2w, part_translation, part_rotation):
    x_min, y_min, z_min = min_bound
    x_max, y_max, z_max = max_bound
    points = []
    # Get the eight corner points
    points.append([x_min, y_min, z_min])
    points.append([x_max, y_min, z_min])
    points.append([x_max, y_max, z_min])
    points.append([x_min, y_max, z_min])
    points.append([x_min, y_min, z_max])
    points.append([x_max, y_min, z_max])
    points.append([x_max, y_max, z_max])
    points.append([x_min, y_max, z_max])
    # Get 12 edge midpoints
    points.append([x_min, y_min, 0])
    points.append([x_max, y_min, 0])
    points.append([x_max, y_max, 0])
    points.append([x_min, y_max, 0])
    points.append([x_min, 0, z_min])
    points.append([x_max, 0, z_min])
    points.append([0, y_min, z_min])
    points.append([0, y_max, z_min])
    points.append([x_min, 0, z_max])
    points.append([x_max, 0, z_max])
    points.append([0, y_min, z_max])
    points.append([0, y_max, z_max])
    # Get 6 face mid points
    points.append([0, 0, z_min])
    points.append([0, 0, z_max])
    points.append([x_max, 0, 0])
    points.append([x_min, 0, 0])
    points.append([0, y_min, 0])
    points.append([0, y_max, 0])

    pose_transformation = np.eye(4)
    pose_transformation[0:3, 3] = part_translation
    pose_transformation[0:3, 0:3] = eulerAnglesToRotationMatrix(part_rotation)
    to_world = np.dot(c2w, pose_transformation)
    points = np.transpose(np.array([point + [1] for point in points]))
    points = np.dot(to_world, points)
    points = np.transpose(points)[:, 0:3]
    points = [tuple(point) for point in points.tolist()]

    return points


def getAxisLineDistance(candidate_points, origin_world, axis_world):
    candidate_points = np.array(candidate_points)
    test = candidate_points - origin_world
    distance = np.linalg.norm(np.cross(test, axis_world), axis=1) / np.linalg.norm(axis_world)
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

    traintest = ["train", "valid", "test"]
    all_parts = {}

    for mode in traintest:
        if mode not in all_parts:
            all_parts[mode] = {}
        for model_path in model_paths:
            model_name = model_path.split('/')[-1]
            if mode == "train":
                if model_name in test_ids or model_name in valid_ids:
                    continue
            elif mode == "valid":
                if model_name not in valid_ids:
                    continue
            else:
                if model_name not in test_ids:
                    continue
            # Only open the close state image (one image contains all annotaions for the joints)
            json_path = model_path + f"/origin_annotation/{model_name}-0-0.json"

            json_file = open(json_path)
            annotation = json.load(json_file)
            json_file.close()

            motions = annotation["motions"]
            # extrinsic is the transformation matrix from camera coordinate to the world coordinate
            c2w = np.reshape(annotation["camera"]["extrinsic"]["matrix"], (4, 4)).T

            for motion in motions:
                part_label = motion["label"]
                motion_type = motion["type"]
                origin_cam = motion["current_origin"]
                axis_cam = motion["current_axis"]

                origin_world = np.dot(c2w, np.array(origin_cam + [1]))[0:3]
                axis_end_cam = list(np.array(axis_cam) + np.array(origin_cam))
                axis_end_world = np.dot(c2w, np.array(axis_end_cam + [1]))[0:3]
                axis_world = axis_end_world - origin_world

                if part_label not in all_parts[mode]:
                    all_parts[mode][part_label] = {
                        "motion_type": [],
                        "axis_world": [],
                        "origin_world": [],
                        "part_pose": [],
                        "c2w": [],
                    }

                all_parts[mode][part_label]["motion_type"].append(motion_type)
                all_parts[mode][part_label]["origin_world"].append(origin_world)
                all_parts[mode][part_label]["axis_world"].append(axis_world)

                all_parts[mode][part_label]["part_pose"].append(motion["partPose"])
                all_parts[mode][part_label]["c2w"].append(c2w)

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
            # Find the most frequent motion origin
            # Find all candidate part points
            candidate_points = [(0, 0, 0)]
            for c2w, part_pose in zip(
                all_parts[mode][part_label]["c2w"], all_parts[mode][part_label]["part_pose"]
            ):
                part_dimension = np.array(part_pose["dimension"])
                part_translation = np.array(part_pose["translation"])
                part_rotation = np.array(part_pose["rotation"])

                min_bound = (
                    -part_dimension[0] / 2,
                    -part_dimension[1] / 2,
                    -part_dimension[2] / 2,
                )
                max_bound = (
                    part_dimension[0] / 2,
                    part_dimension[1] / 2,
                    part_dimension[2] / 2,
                )
                # add potential points
                points = getALLPoints(
                    min_bound, max_bound, c2w, part_translation, part_rotation
                )
                candidate_points += points
            candidate_points = set(candidate_points)
            candidate_points = list(candidate_points)

            candidate_count = [0] * len(candidate_points)
            print(len(all_parts[mode][part_label]["motion_type"]))
            for axis_world, origin_world in zip(
                all_parts[mode][part_label]["axis_world"], all_parts[mode][part_label]["origin_world"]
            ):
                distance = getAxisLineDistance(candidate_points, origin_world, axis_world)
                candidate_count[np.argmin(distance)] += 1
            statistics[mode][part_label]["origin_world"] = list(candidate_points[np.argmax(candidate_count)])

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
            statistics[mode][part_label]["axis_world"] = base_axis[np.argmax(axis_number)].tolist()

    print(statistics)

    most_frequent_file = open('/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/data_statistics/most_frequent.json', 'w')
    json.dump(statistics, most_frequent_file)
    most_frequent_file.close()

    stop = time()
    print(str(stop - start) + " seconds")
