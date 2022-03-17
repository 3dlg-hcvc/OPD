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

RAWDATAPATH = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/raw_data_6.0/"

# Use background from corresponding directory
TESTIDSPATH = './testIds.json'
VALIDIDPATH = './validIds.json'

bug_model = ["102192", "103013", "102182", "102173", "103012", "12617", "103369", "103425", "102193", "103633", "102418", "102177", "102163", "12259", "103452", "100283", "102234", "103351", "12587", "102301", "102252"]

total_number = 0

def processImage(image_path, model_name):
    global total_number
    img_name = (image_path.split("/")[-1]).split(".")[0]
    

    depth_file_path = f"{RAWDATAPATH}{model_name}/depth/{img_name}_d.png"
    depth = np.array(Image.open(depth_file_path))
    # Copy all mask images (One origin <-> Multiple masks)
    mask_file_paths = glob.glob(f"{RAWDATAPATH}{model_name}/mask/{img_name}_*")
    

    for mask_file_path in mask_file_paths:
        mask = Image.open(mask_file_path)
        mask = np.array(mask)

        for i in range(np.shape(mask)[0]):
            for j in range(np.shape(mask)[1]):
                if(mask[i, j, 3] > 1) and depth[i, j] == 0:
                    total_number += 1
                    return True
                
    return False


if __name__ == "__main__":
    start = time()
    # Load the model in train/test/valid
    test_ids_file = open(TESTIDSPATH)
    test_ids = json.load(test_ids_file)
    test_ids_file.close()

    valid_ids_file = open(VALIDIDPATH)
    valid_ids = json.load(valid_ids_file)
    valid_ids_file.close()

    print(f"bug models {len(bug_model)}")

    # Deal with the images in raw dataset
    model_paths = glob.glob(RAWDATAPATH + "*")
    for model_path in model_paths:
        model_name = model_path.split("/")[-1]
        # if not model_name in bug_model:
        #     continue 

        if model_name in test_ids:
            subdir = 'test'
        elif model_name in valid_ids:
            subdir = 'valid'
        else:
            subdir = 'train'
        
        flag = False
        
        image_paths = glob.glob(model_path + "/origin/*")
        total_number += len(image_paths)
        # for image_path in image_paths:
        #     # pool.apply_async(processImage, (image_path, model_name,))
        #     flag = processImage(image_path, model_name)
        #     # if flag == True:
        #     #     break

        if flag == True:
            print(f"DOUBLE FACE ERROR: {model_name} in {subdir}")

    print(f"total number {total_number}")
    
    stop = time()
    print(str(stop - start) + " seconds")
