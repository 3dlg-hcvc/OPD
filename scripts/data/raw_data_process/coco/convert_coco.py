import json
import os
import argparse
import numpy as np
from pycococreatortools import pycococreatortools
import multiprocessing 
from PIL import Image
import json

PROCESSDATAPATH = '/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/process_data_real/'

def existDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

# the map for image name and id
img_name_id = {}
motion_annotations = {}
cat_id = {}

# Unuseful Information
def get_info():

    info = {
        "description": "MotionNet",
        "url": "",
        "version": "0.0.0",
        "year": 2020,
        "contributor": "Shawn, Angel",
        "date_created": "2020/07/04"
    }
    return info

# Unuseful Information
def get_licenses():
    licenses = [
    {
        "url": "",
        "id": 1,
        "name": ""
    }]

    return licenses

# MotionNet Version
def get_categories():
    global cat_id

    categories = []

    # Don't make use of super category here
    supercategory = ['Container_parts']

    # Drawer
    info = {}
    info["supercategory"] = supercategory[0]
    info["id"] = 1
    info["name"] = "drawer"
    categories.append(info)
    cat_id['drawer'] = 1
    # Door
    info = {}
    info["supercategory"] = supercategory[0]
    info["id"] = 2
    info["name"] = "door"
    categories.append(info)
    cat_id['door'] = 2
    # Lid
    info = {}
    info["supercategory"] = supercategory[0]
    info["id"] = 3
    info["name"] = "lid"
    categories.append(info)
    cat_id['lid'] = 3

    return categories

# MotionNet Version
def get_images(img_folder, annotation_folder):
    global img_name_id
    global motion_annotations

    images = []

    img_id = 1
    for img_file_name in os.listdir(img_folder):
        img_name = img_file_name.split('.')[0]

        # Read corresponding annotation file
        annotation_file = open(f'{annotation_folder}/{img_name}.json')
        data = json.load(annotation_file)
        annotation_file.close()
        
        img_path = os.path.join(img_folder, img_file_name)

        image = {
                "license": 1,
                "file_name": img_file_name,
                "coco_url" :"",
                "height": data['height'],
                "width" : data['width'],
                "date_captured": "",
                "flickr_url": "",
                "id": img_id
            }

        img_name_id[img_name] = img_id

        # MotionNet Extra Annotation
        # Add corresponding depth image name
        image['depth_file_name'] = img_name + '_d.png'
        # Add camera parameters to the image
        image['camera'] = data['camera']
        # Add model label
        image['label'] = data['label']

        images.append(image)

        # Preprocess the motion annotations for adding it into the masks
        motion_annotations[img_name] = {}
        for motion in data['motions']:
            motion_annotations[img_name][motion['partId']] = motion

        img_id += 1

    return images

# MotionNet Version
def get_annotation_info(seg_id, img_id, category_info, mask_path, motion):
    
    mask = Image.open(mask_path)
    binary_mask = np.asarray(mask.convert('L')).astype(np.uint8)

    for i in range(np.shape(binary_mask)[0]):
        for j in range(np.shape(binary_mask)[1]):
            if(binary_mask[i, j] > 1):
                binary_mask[i, j] = 1
    
    annotation_info = pycococreatortools.create_annotation_info(
                        seg_id, img_id, category_info, binary_mask,
                        mask.size, tolerance=2)

    if(annotation_info != None):
        annotation_info['motion'] = motion

    return annotation_info

# MotionNet Version
def get_annotations(img_folder, mask_folder):
    global img_name_id
    global motion_annotations
    global cat_id

    annotations = []
    
    pool = multiprocessing.Pool(processes = 16)

    seg_id = 1
    annotation_infos = []

    for mask_name in os.listdir(mask_folder):
        img_name = mask_name.rsplit('_', 1)[0]
        if img_name in img_name_id.keys():
            img_id = img_name_id[img_name]
        else:
            print('ERROR: cannot find image ', img_name)
            exit()

        part_id = (mask_name.split('.')[0]).split('_')[-1]

        motion = motion_annotations[img_name][part_id]
        if motion['label'].strip() not in cat_id:
            print(motion['label'].strip())
        part_cat = cat_id[motion['label'].strip()]
        
        category_info = {'id': part_cat, 'is_crowd': 0}
        mask_path = os.path.join(mask_folder, mask_name)

        annotation_infos.append(pool.apply_async(get_annotation_info, (seg_id, img_id, category_info, mask_path, motion, )))

        seg_id += 1
    
    pool.close()
    pool.join()

    for i in annotation_infos:
        annotation_info = i.get()
        if(annotation_info != None):
            annotations.append(annotation_info)

    return annotations

def save_json(data, save_path):
    out_json = json.dumps(data, sort_keys=True, indent=4, separators=(',', ':'),
                          ensure_ascii=False)
    fo = open(save_path, "w")
    fo.write(out_json)
    fo.close()

if __name__ == "__main__":

    annotation_data_path = PROCESSDATAPATH + 'coco_annotation/'
    existDir(annotation_data_path)

    # Deal with train data
    print('process train data ...')
    train_origin_path = PROCESSDATAPATH + 'train/origin'
    train_mask_path = PROCESSDATAPATH + 'train/mask'
    train_annotation_path = PROCESSDATAPATH + 'train/origin_annotation'

    output = {}
    output["info"] = get_info()
    output["licenses"] = get_licenses()
    output["categories"] = get_categories()
    output["images"] = get_images(train_origin_path, train_annotation_path)
    output["annotations"] = get_annotations(train_origin_path, train_mask_path)

    save_json(output, annotation_data_path + 'MotionNet_train.json')

    # Deal with valid data
    # Reset global variables
    img_name_id = {}
    motion_annotations = {}
    cat_id = {}

    print('process valid data ...')
    valid_origin_path = PROCESSDATAPATH + 'valid/origin'
    valid_mask_path = PROCESSDATAPATH + 'valid/mask'
    valid_annotation_path = PROCESSDATAPATH + 'valid/origin_annotation'

    output = {}
    output["info"] = get_info()
    output["licenses"] = get_licenses()
    output["categories"] = get_categories()
    output["images"] = get_images(valid_origin_path, valid_annotation_path)
    output["annotations"] = get_annotations(valid_origin_path, valid_mask_path)

    save_json(output, annotation_data_path + 'MotionNet_valid.json')

    # Deal with test data
    # Reset global variables
    img_name_id = {}
    motion_annotations = {}
    cat_id = {}

    print('process test data ...')
    test_origin_path = PROCESSDATAPATH + 'test/origin'
    test_mask_path = PROCESSDATAPATH + 'test/mask'
    test_annotation_path = PROCESSDATAPATH + 'test/origin_annotation'

    output = {}
    output["info"] = get_info()
    output["licenses"] = get_licenses()
    output["categories"] = get_categories()
    output["images"] = get_images(test_origin_path, test_annotation_path)
    output["annotations"] = get_annotations(test_origin_path, test_mask_path)

    save_json(output, annotation_data_path + 'MotionNet_test.json')