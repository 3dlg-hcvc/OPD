from time import time
from PIL import Image
import cv2
import glob
import os
import numpy as np
import h5py
import json
from collections import OrderedDict
import math
import yaml
import multiprocessing
import json

# RAWDATAPATH = "/home/hja40/Desktop/Research/proj-motionnet/Dataset/raw_data_6.0/"
RAWDATAPATH = "/home/hja40/Desktop/Research/proj-motionnet/Dataset/raw_data_articulated_pose"
OUTPUTPATH = "/home/hja40/Desktop/Research/articulated-pose/dataset/sapien/render/drawer"

def breakpoint():
    import pdb

    pdb.set_trace()


def existDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def getConventionTransform(source):
    transformation = np.matrix(np.eye(4))
    if source == "partnetsim" or source == "sapien" or source == "SAPIEN":
        transformation[0, 0] = -1
        transformation[2, 2] = -1
    elif source == "shape2motion":
        transformation[0:3, 0:3] = np.matrix([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    return transformation.I


def getFocalLength(FOV, height, width=None):
    # FOV is in radius, should be vertical angle
    if width == None:
        f = height / (2 * math.tan(FOV / 2))
        return f
    else:
        fx = height / (2 * math.tan(FOV / 2))
        fy = fx / height * width
        return (fx, fy)


def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4,))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array(
            [
                [m00 - m11 - m22, 0.0, 0.0, 0.0],
                [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
            ]
        )
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


# Convert the transformation matrix into translation and rotation in quaternion ([w, x, y, z])
def parseMatrix(matrix):
    pos = list(matrix[:3, 3])
    rot = matrix[:3, :3]

    orn = quaternion_from_matrix(rot)
    return pos, orn


# Process the annotations
def processAnno(dir_path, poses, pose_map, valid_images, part_indexes):
    model_id = dir_path.split("/")[-1]
    for pose in poses.keys():
        image_names = poses[pose]
        yml_dict = OrderedDict()
        image_num = 0
        for image_name in image_names:
            if image_name not in valid_images:
                continue
            viewpoint_index = image_name[-1]
            json_file = open(f"{dir_path}/origin_annotation/{image_name}.json")
            annotation = json.load(json_file)
            json_file.close()
            # Prepare the viewMatrix
            camera2world = np.reshape(
                annotation["camera"]["extrinsic"]["matrix"], (4, 4)
            ).T
            world2camera = np.array(np.mat(camera2world).I)
            # flat through column
            viewMatrix = list(world2camera.flatten("F"))

            # Prepare the projection matrix
            img_height = 512
            img_width = 512
            FOV = annotation["camera"]["intrinsic"]["fov"]
            fx, fy = getFocalLength(FOV / 180 * math.pi, img_height, img_width)
            cy = img_height / 2
            cx = img_width / 2
            projMatrix = [
                fx,
                0,
                np.NaN,
                0,
                0,
                -fy,
                np.NaN,
                0,
                -cx,
                -cy,
                np.NaN,
                -1,
                0,
                0,
                np.NaN,
                0,
            ]

            # Prepare the link state
            initial_pose = np.array(getConventionTransform(annotation["source"]))
            current_poses = {}
            for motion in annotation["motions"]:
                part_index = int(motion["partId"])
                current_pose = np.reshape(motion["world"], (4, 4)).T
                current_poses[part_index] = current_pose

            if image_num == 0:
                # The poses are the same for different viewpoints (this is the model2world pose)
                link_poses = OrderedDict()
                for part_index in range(part_indexes[image_name] + 1):
                    if part_index in current_poses.keys():
                        # This part is a moving part, has differet pose other than the initial pose
                        pos, orn = parseMatrix(current_poses[part_index])
                    else:
                        pos, orn = parseMatrix(initial_pose)
                    link_poses[part_index] = OrderedDict(
                        [
                            (0, np.NaN),
                            (1, np.NaN),
                            (2, np.NaN),
                            (3, np.NaN),
                            (4, pos),
                            (5, orn),
                        ]
                    )
            yml_dict["frame_{}".format(viewpoint_index)] = OrderedDict(
                [
                    ("obj", link_poses),
                    ("viewMat", list(viewMatrix)),
                    ("projMat", list(projMatrix)),
                ]
            )
            image_num += 1
        with open(f"{OUTPUTPATH}/{model_id}/{pose_map[pose]}/gt.yml", 'w') as f:
            yaml.dump(yml_dict, f, default_flow_style=False)


# Process the mask images
def processMask(dir_path, poses, pose_map, image_map):
    model_id = dir_path.split("/")[-1]
    valid_images = []
    part_indexes = {}
    for pose in poses.keys():
        image_names = poses[pose]
        for image_name in image_names:
            viewpoint_index = image_name[-1]
            existDir(f"{OUTPUTPATH}/{model_id}/{pose_map[pose]}/mask")
            mask_paths = glob.glob(f"{dir_path}/mask_512/{image_name}*.png")
            new_mask = np.ones((512, 512)) * 255
            is_valid = True
            max_index = 0
            for mask_path in mask_paths:
                part_index = int(mask_path.split(".")[0][-1])
                max_index = max(max_index, part_index)
                mask = Image.open(mask_path)
                binary_mask = (np.asarray(mask.convert("L")).astype(np.uint8)) > 1
                if np.sum(binary_mask) < 10:
                    is_valid = False
                    break
                new_mask[binary_mask] = part_index + 1

            if is_valid == False:
                print("Invalid Image: ", image_name)
                continue
            else:
                part_indexes[image_name] = max_index
                valid_images.append(image_name)
                # im = Image.fromarray(new_mask*50)
                # im.show()
                # print(np.unique(new_mask))
                image_map[f'{model_id}_{pose_map[pose]}_{viewpoint_index}'] = image_name
                cv2.imwrite(
                    f"{OUTPUTPATH}/{model_id}/{pose_map[pose]}/mask/{viewpoint_index}.png",
                    new_mask,
                )
    return (valid_images, part_indexes)


# Process the depth images
def processDepth(dir_path, poses, pose_map, valid_images):
    model_id = dir_path.split("/")[-1]
    for pose in poses.keys():
        image_names = poses[pose]
        for image_name in image_names:
            if image_name not in valid_images:
                continue
            # print(image_name)
            viewpoint_index = image_name[-1]
            existDir(f"{OUTPUTPATH}/{model_id}/{pose_map[pose]}/depth")
            image = Image.open(f"{dir_path}/depth_512/{image_name}_d.png")
            depth = np.array(image) / 1000
            depth_name = (
                f"{OUTPUTPATH}/{model_id}/{pose_map[pose]}/depth/{viewpoint_index}.h5"
            )
            hf = h5py.File(depth_name, "w")
            hf.create_dataset("data", data=depth)


# Process the origin images
def processOrigin(dir_path, poses, pose_map, valid_images):
    model_id = dir_path.split("/")[-1]
    for pose in poses.keys():
        image_names = poses[pose]
        for image_name in image_names:
            if image_name not in valid_images:
                continue
            # print(image_name)
            viewpoint_index = image_name[-1]
            existDir(f"{OUTPUTPATH}/{model_id}/{pose_map[pose]}/rgb")
            image = Image.open(f"{dir_path}/origin/{image_name}.png")
            new_image = image.resize((512, 512))
            new_image.save(
                f"{OUTPUTPATH}/{model_id}/{pose_map[pose]}/rgb/{viewpoint_index}.png"
            )


# Convert the raw dataset (something like raw_dataset 6.0) into the articulated pose dataset format (Only for the structure)
def convert(dir_path, image_map):
    model_id = dir_path.split("/")[-1]
    existDir(f"{OUTPUTPATH}/{model_id}")
    model_paths = glob.glob(f"{dir_path}/origin/*")
    poses = {}
    pose_map = {}
    pose_index = 0
    for model_path in model_paths:
        image_name = model_path.split("/")[-1].split(".")[0]
        index1 = image_name.find("-")
        pose_name = image_name[index1 + 1 : -2]
        if pose_name not in poses:
            poses[pose_name] = [image_name]
            pose_map[pose_name] = pose_index
            existDir(f"{OUTPUTPATH}/{model_id}/{pose_index}")
            pose_index += 1
        else:
            poses[pose_name].append(image_name)
        # viewpoint = image_name[-1]
        # print(image_name[index1+1:-2])

    # print(pose_map)
    # valid_images = []

    # Deal with the mask images
    valid_images, part_indexes = processMask(dir_path, poses, pose_map, image_map)

    #  Deal with the origin images
    processOrigin(dir_path, poses, pose_map, valid_images)

    # Deal with the depth images
    processDepth(dir_path, poses, pose_map, valid_images)

    #  Deal with the annotations (viewMat, projMat, obj -> joint state)
    processAnno(dir_path, poses, pose_map, valid_images, part_indexes)

    # print(poses)


if __name__ == "__main__":
    start = time()

    model_list = ['40453', '44962', '45132',
                    '45290', '46130', '46334',  '46462',
                    '46537', '46544', '46641', '47178', '47183',
                    '47296', '47233', '48010', '48253',  '48517',
                    '48740', '48876', '46230', '44853', '45135',
                    '45427', '45756', '46653', '46879', '47438', '47711', '48491', '46123',  '45841', '46440']
    
    pool = multiprocessing.Pool(processes=16)

    image_map = multiprocessing.Manager().dict()

    for model_id in model_list:
        print(f"Processing Model {model_id}")
        model_path = f"{RAWDATAPATH}/{model_id}"
        # convert(model_path)
        pool.apply_async(convert, (model_path, image_map,))

    pool.close()
    pool.join()

    # print(image_map)
    json_file = open(f'{RAWDATAPATH}/image_map.json', 'w')
    json.dump(dict(image_map), json_file)
    json_file.close()


    stop = time()
    print(str(stop - start) + " seconds")
