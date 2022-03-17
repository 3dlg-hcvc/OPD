# This script is optional, should be after raw_preprocess.py and before split.py
from PIL import Image
import numpy as np
import glob
import os
import random
import multiprocessing
from time import time
import sys, traceback
import json

RAWDATAPATH = "../../../Dataset/raw_data_6.0/"
BACKGROUND = "../../../Dataset/background/RGB/"

OUPUTPATH = "../../../Dataset/raw_data_6.1/"
NUM_BG = 4

# Use background from corresponding directory
TESTIDSPATH = './testIds.json'
VALIDIDPATH = './validIds.json'

def existDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def copyFile(old_path, new_path):
    os.system(f"cp {old_path} {new_path}")


def processImage(image_path, model_name, backgrounds):
    # Copy the origin image, mask, depth and annotations (depth doesn't contain the background)
    img_name = (image_path.split("/")[-1]).split(".")[0]
    # Copy the origin image
    origin_file_path = f"{RAWDATAPATH}{model_name}/origin/{img_name}.png"
    new_origin_path = f"{OUPUTPATH}{model_name}/origin/{img_name}.png"
    copyFile(origin_file_path, new_origin_path)
    # Copy the depth image (One origin <-> One depth)
    depth_file_path = f"{RAWDATAPATH}{model_name}/depth/{img_name}_d.png"
    new_depth_path = f"{OUPUTPATH}{model_name}/depth/{img_name}_d.png"
    copyFile(depth_file_path, new_depth_path)
    # Copy all mask images (One origin <-> Multiple masks)
    mask_file_paths = glob.glob(f"{RAWDATAPATH}{model_name}/mask/{img_name}_*")
    for mask_file_path in mask_file_paths:
        mask_file_name = mask_file_path.split("/")[-1]
        new_mask_path = f"{OUPUTPATH}{model_name}/mask/{mask_file_name}"
        copyFile(mask_file_path, new_mask_path)
    # Copy the annotation (One origin <-> One annotation)
    anno_file_path = f"{RAWDATAPATH}{model_name}/origin_annotation/{img_name}.json"
    new_anno_path = f"{OUPUTPATH}{model_name}/origin_annotation/{img_name}.json"
    copyFile(anno_file_path, new_anno_path)
    # Get four images with random backgroud
    try:
        for k in range(len(backgrounds)):
            # Judge if it's background through the alpha channel
            RGB = np.array(Image.open(origin_file_path))
            bg = np.array(Image.open(backgrounds[k]))
            h, w, c = np.shape(RGB)

            for i in range(h):
                for j in range(w):
                    if RGB[i][j][3] != 255:
                        RGB[i][j][3] = 255
                        RGB[i][j][0:3] = bg[i][j][0:3]
            image = Image.fromarray(RGB)
            # Store the new image and corresponding masks, depth and annotation
            new_image_name = f"{img_name}+bg{k}"
            image.save(f"{OUPUTPATH}{model_name}/origin/{new_image_name}.png")
            # Copy the depth image (One origin <-> One depth)
            new_depth_path = f"{OUPUTPATH}{model_name}/depth/{new_image_name}_d.png"
            copyFile(depth_file_path, new_depth_path)
            # Copy all mask images (One origin <-> Multiple masks)
            for mask_file_path in mask_file_paths:
                mask_number = (mask_file_path.split("/")[-1]).split("_")[-1]
                new_mask_path = (
                    f"{OUPUTPATH}{model_name}/mask/{new_image_name}_{mask_number}"
                )
                copyFile(mask_file_path, new_mask_path)
            # Copy the annotation (One origin <-> One annotation)
            new_anno_path = (
                f"{OUPUTPATH}{model_name}/origin_annotation/{new_image_name}.json"
            )
            copyFile(anno_file_path, new_anno_path)
    except:
        traceback.print_exc(file=sys.stdout)
    


if __name__ == "__main__":
    start = time()
    # Load the model in train/test/valid
    test_ids_file = open(TESTIDSPATH)
    test_ids = json.load(test_ids_file)
    test_ids_file.close()

    valid_ids_file = open(VALIDIDPATH)
    valid_ids = json.load(valid_ids_file)
    valid_ids_file.close()

    # Get the background list
    backgrounds = {}
    backgrounds['train'] = glob.glob(f"{BACKGROUND}train/*")
    backgrounds['test'] = glob.glob(f"{BACKGROUND}test/*")
    backgrounds['valid'] = glob.glob(f"{BACKGROUND}valid/*")
    pool = multiprocessing.Pool(processes=16)
    # Deal with the images in raw dataset
    model_paths = glob.glob(RAWDATAPATH + "*")
    for model_path in model_paths:
        model_name = model_path.split("/")[-1]
        print(model_name)
        new_path = f"{OUPUTPATH}{model_name}/"
        existDir(f"{new_path}origin/")
        existDir(f"{new_path}depth/")
        existDir(f"{new_path}mask/")
        existDir(f"{new_path}origin_annotation/")

        if model_name in test_ids:
            subdir = 'test'
        elif model_name in valid_ids:
            subdir = 'valid'
        else:
            subdir = 'train'
        image_paths = glob.glob(model_path + "/origin/*")
        for image_path in image_paths:
            # Get random backgrounds
            bgs = []
            for i in range(NUM_BG):
                bgs.append(backgrounds[subdir][random.randint(0, len(backgrounds[subdir]) - 1)])
            pool.apply_async(processImage, (image_path, model_name, bgs,))
    pool.close()
    pool.join()

    stop = time()
    print(str(stop - start) + " seconds")
