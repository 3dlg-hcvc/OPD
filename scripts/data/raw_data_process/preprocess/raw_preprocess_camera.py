from PIL import Image
import glob
import json
import numpy as np
import multiprocessing
from time import time
import math

RAWDATAPATH = "/local-scratch/localhome/hja40/Desktop/Dataset/raw_data_6.2/"


def getConventionTransform(source):
    transformation = np.matrix(np.eye(4))
    if source == "partnetsim" or source == "sapien" or source == "SAPIEN":
        transformation[0, 0] = -1
        transformation[2, 2] = -1
    elif source == "shape2motion":
        transformation[0:3, 0:3] = np.matrix([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    return transformation.I


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The reuslt is for euler angles (ZYX) radians
def rotationMatrixToEulerAngles(R):

    assert isRotationMatrix(R)

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


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
    extrinsic = np.reshape(annotation["camera"]["extrinsic"]["matrix"], (4, 4)).T
    w2c = np.array(np.mat(extrinsic).I)

    for motion in motions:
        current_pose = np.reshape(motion["world"], (4, 4)).T
        # print(current_pose)
        # print(motion['axis'])
        motion.pop("world")
        initial_pose = np.array(getConventionTransform(annotation["source"]))

        """ Deal with the part pose """
        # Store the 3DOF dimension (Because the convention is also in the same axis, then the raw annotation is okay)
        min_bound_raw = motion["3dbbx"]["min"]
        max_bound_raw = motion["3dbbx"]["max"]
        x_min, y_min, z_min = min_bound_raw["x"], min_bound_raw["y"], min_bound_raw["z"]
        x_max, y_max, z_max = max_bound_raw["x"], max_bound_raw["y"], max_bound_raw["z"]
        pose_dimension = [x_max - x_min, y_max - y_min, z_max - z_min]
        if pose_dimension[0] < 0 or pose_dimension[1] < 0 or pose_dimension[2] < 0:
            raise ValueError(
                "The dimension is negative. In normal case, this shouldn't happen"
            )

        # Calculate the transformation matrix from consistent pose in camera coordinate to the current pose in camera coordinate
        # The transformation is composed by three parts: consistent2initial, initial2current, w2c
        ## Calculate the translation from the consistent pose to the initial pose
        bbx_center_raw = np.array(
            [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2, 1]
        )
        bbx_center_world = np.dot(initial_pose, bbx_center_raw)[0:3]
        consistent2initial = np.eye(4)
        consistent2initial[0:3, 3] = bbx_center_world
        ## Calculate the transformation from initial world pose to current world pose
        initial2current = np.dot(current_pose, np.array(np.mat(initial_pose).I))
        ## Get the final transformation
        pose_transformation = np.dot(w2c, np.dot(initial2current, consistent2initial))
        # Extract 3DOF translation and 3DOF rotation
        pose_translation = list(pose_transformation[0:3, 3])
        pose_rotation = pose_transformation[0:3, 0:3]
        pose_euler = list(rotationMatrixToEulerAngles(pose_rotation))
        motion["partPose"] = {
            "dimension": pose_dimension,
            "translation": pose_translation,
            "rotation": pose_euler,
        }
        motion.pop("3dbbx")

        """ Convert the motion annotations into camera coordiante """
        current_pose = np.dot(w2c, current_pose)
        initial_pose = np.dot(w2c, initial_pose)

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
            # processJson(json_path)
            # break
            pool.apply_async(processJson, (json_path,))

    pool.close()
    pool.join()

    stop = time()
    print(str(stop - start) + " seconds")

