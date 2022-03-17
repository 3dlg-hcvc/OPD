import json
import glob
from PIL import Image
import numpy as np

mode = "real"

if not mode == "real":
    TESTIDSPATH = '/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/raw_data_process/preprocess/testIds.json'
    VALIDIDPATH = '/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/raw_data_process/preprocess/validIds.json'
    RAWDATAPATH = "/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/raw_data_6.1/"
else:
    TESTIDSPATH = '/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/raw_data_process/preprocess/testIds_real.json'
    VALIDIDPATH = '/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/raw_data_process/preprocess/validIds_real.json'
    RAWDATAPATH = "/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/raw_data_real/"

def processModel(model):
    result = {"drawer": 0, "door": 0, "lid": 0, "translation": 0, "rotation": 0}
    annotation_paths = glob.glob(f"{model}/origin_annotation/*")

    for annotation_path in annotation_paths:
        anno_file = open(annotation_path)
        annotation = json.load(anno_file)["motions"]
        anno_file.close()

        annotation_name = annotation_path.split('/')[-1].split('.')[0]
        mask_paths = glob.glob(f"{model}/mask/{annotation_name}_*.png")

        valid_motion_index = []
        for mask_path in mask_paths:
            raw_mask = np.array(Image.open(mask_path))
            if not mode == "real":
                num_valid = np.where(raw_mask[:, :, 1] == 255)[0].shape[0]
            else:
                num_valid = np.where(raw_mask == True)[0].shape[0]

            if num_valid > 0:
                valid_motion_index.append(mask_path.rsplit('.', 1)[0].rsplit('_', 1)[-1])
        
        for part in annotation:
            if part["partId"] not in valid_motion_index:
                continue
            if part["label"].strip() not in result.keys():
                print("Label Error")
                import pdb
                pdb.set_trace()
            result[part["label"].strip()] += 1

            if part["type"].strip() not in result.keys():
                print("Type Error")
                import pdb
                pdb.set_trace()
            result[part["type"].strip()] += 1

    return result

if __name__ == "__main__":
    test_ids_file = open(TESTIDSPATH)
    test_ids = json.load(test_ids_file)
    test_ids_file.close()

    valid_ids_file = open(VALIDIDPATH)
    valid_ids = json.load(valid_ids_file)
    valid_ids_file.close()

    models = glob.glob(f"{RAWDATAPATH}/*")
    part_labels = {"train": {"drawer": 0, "door": 0, "lid": 0, "translation": 0, "rotation": 0}, "val": {"drawer": 0, "door": 0, "lid": 0, "translation": 0, "rotation": 0}, "test": {"drawer": 0, "door": 0, "lid": 0, "translation": 0, "rotation": 0}}
    for model in models:
        model_name = model.split('/')[-1]
        if model_name in valid_ids:
            set_name = "val"
        elif model_name in test_ids:
            set_name = "test"
        else:
            set_name = "train"
        result = processModel(model)
        part_labels[set_name]["drawer"] += result["drawer"]
        part_labels[set_name]["door"] += result["door"]
        part_labels[set_name]["lid"] += result["lid"]
        part_labels[set_name]["translation"] += result["translation"]
        part_labels[set_name]["rotation"] += result["rotation"]
    
    if mode == "real":
        output_path =  "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/data_statistics/real_part_label.json"
    else:
        output_path = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/data_statistics/part_label.json"
    part_label_file = open(
        output_path,
        "w",
    )
    json.dump(part_labels, part_label_file)
    part_label_file.close()
