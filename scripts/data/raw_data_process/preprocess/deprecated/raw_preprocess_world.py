from PIL import Image
import glob
import json
import numpy as np
import multiprocessing
from time import time

RAWDATAPATH = "../../../Dataset/raw_data_4.0/"


def getConventionTransform(source):
    transformation = np.matrix(np.eye(4))
    if source == "partnetsim" or source == "sapien" or source == "SAPIEN":
        transformation[0, 0] = -1
        transformation[2, 2] = -1
    elif source == "shape2motion":
        transformation[0:3, 0:3] = np.matrix([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    return transformation.I


def getPoints(min_bound, max_bound):
    # This funciton is for axis-aligned bbx
    x_min, y_min, z_min = min_bound
    x_max, y_max, z_max = max_bound
    # Create the bounding box
    points = []
    points.append([x_min, y_min, z_min])
    points.append([x_max, y_min, z_min])
    points.append([x_max, y_max, z_min])
    points.append([x_min, y_max, z_min])
    points.append([x_min, y_min, z_max])
    points.append([x_max, y_min, z_max])
    points.append([x_max, y_max, z_max])
    points.append([x_min, y_max, z_max])
    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    return (points, lines)


def processBBX(bbx, transformation):
    # The bounding box in the raw annotation is an axis-aligned bbx
    min_bound_raw = bbx["min"]
    max_bound_raw = bbx["max"]
    min_bound = np.array(
        [min_bound_raw["x"], min_bound_raw["y"], min_bound_raw["z"]]
    ).astype(np.float32)
    max_bound = np.array(
        [max_bound_raw["x"], max_bound_raw["y"], max_bound_raw["z"]]
    ).astype(np.float32)
    # Get the eight points and corresponding lines
    points, lines = getPoints(min_bound, max_bound)
    # transform the points
    new_points = []
    for point in points:
        point = np.array(point + [1])
        new_point = np.dot(transformation, point)[0:3]
        new_points.append(list(new_point))
    return {"points": new_points, "lines": lines}


def processOrigin(origin_raw, transformation):
    origin = np.array([origin_raw["x"], origin_raw["y"], origin_raw["z"]] + [1]).astype(
        np.float32
    )
    new_origin = np.dot(transformation, origin)[0:3]
    return list(new_origin)


def processAxis(axis_raw, origin_raw, transformation):
    axis = np.array([axis_raw["x"], axis_raw["y"], axis_raw["z"]] + [1]).astype(
        np.float64
    )
    # Use helper point to calculate the new axis
    origin = np.array([origin_raw["x"], origin_raw["y"], origin_raw["z"]] + [1]).astype(
        np.float64
    )
    helper = origin + axis
    helper[3] = 1
    new_origin = np.dot(transformation, origin)[0:3]
    new_helper = np.dot(transformation, helper)[0:3]
    new_axis = new_helper - new_origin
    return list(new_axis)


def processJson(json_path):
    json_file = open(json_path)
    annotation = json.load(json_file)
    json_file.close()
    motions = annotation["motions"]
    for motion in motions:
        current_pose = np.reshape(motion["world"], (4, 4)).T
        # print(current_pose)
        # print(motion['axis'])
        motion.pop("world")
        initial_pose = np.array(getConventionTransform(annotation["source"]))
        # Deal with the bounding box
        motion["current_3dbbx"] = processBBX(motion["3dbbx"], current_pose)
        motion["initial_3dbbx"] = processBBX(motion["3dbbx"], initial_pose)
        motion.pop("3dbbx")
        # Deal with the origin
        motion["current_origin"] = processOrigin(motion["origin"], current_pose)
        motion["initial_origin"] = processOrigin(motion["origin"], initial_pose)
        # Deal with the axis
        motion["current_axis"] = processAxis(
            motion["axis"], motion["origin"], current_pose
        )
        motion["initial_axis"] = processAxis(
            motion["axis"], motion["origin"], initial_pose
        )
        motion.pop("origin")
        motion.pop("axis")

    json_file = open(json_path, "w")
    json.dump(annotation, json_file)
    json_file.close()


def processImage(image_path):
    image = Image.open(image_path)
    new_image = image.resize((256, 256))
    new_image.save(image_path)


if __name__ == "__main__":
    start = time()
    model_paths = glob.glob(RAWDATAPATH + "*")
    pool = multiprocessing.Pool(processes=16)

    for model_path in model_paths:
        print(model_path)
        # Resize the images
        image_paths = glob.glob(model_path + "/origin/*")
        for image_path in image_paths:
            pool.apply_async(processImage, (image_path,))

        # Process the annotations
        json_paths = glob.glob(model_path + "/origin_annotation/*.json")
        for json_path in json_paths:
            pool.apply_async(processJson, (json_path,))

    pool.close()
    pool.join()

    stop = time()
    print(str(stop - start) + " seconds")

