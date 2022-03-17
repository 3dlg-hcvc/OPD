import json
import glob
import math

raw_data = "/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/raw_data_real"

if __name__ == "__main__":
    catModel = {}

    dirs = glob.glob(f"{raw_data}/*")
    for dir in dirs:
        model_name = dir.split('/')[-1]
        annotation_paths = glob.glob(f"{dir}/origin_annotation/*")
        if len(annotation_paths) == 0:
            import pdb
            pdb.set_trace()
        with open(annotation_paths[0]) as f:
            anno = json.load(f)
        obj_cat = anno["label"]
        image_number = len(glob.glob(f"{dir}/origin/*"))
        # Hardcode to fix some label error
        if obj_cat == "storage_fumiture":
            obj_cat = "storage_furniture"
        if obj_cat in catModel.keys():
            catModel[obj_cat] += image_number
        else:
            catModel[obj_cat] = image_number

    for k, v in catModel.items():
        print(f"{k} has {v} images")
