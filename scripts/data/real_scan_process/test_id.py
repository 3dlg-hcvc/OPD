import json
import glob
import math

raw_data = "/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/raw_data_real"
TESTIDSPATH = '/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/raw_data_process/preprocess/testIds_real.json'
VALPATH = '/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/raw_data_process/preprocess/validIds_real.json'

TESTRATIO = 0.15
VALRATIO = 0.15

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
        # Hardcode to fix some label error
        if obj_cat == "storage_fumiture":
            obj_cat = "storage_furniture"
        if obj_cat in catModel.keys():
            catModel[obj_cat].append(model_name)
        else:
            catModel[obj_cat] = [model_name]

    for k, v in catModel.items():
        print(f"{k} has {len(v)} objects")

    test_ids = []
    val_ids = []

    for model_cat in catModel.keys():
        model_num = len(catModel[model_cat])
        test_model_number = math.ceil(model_num * TESTRATIO)
        val_model_number = math.ceil(model_num * VALRATIO)

        current_test_number = 0
        current_val_number = 0

        for model_id in catModel[model_cat]:
            if(current_test_number < test_model_number):
                test_ids.append(model_id)
                current_test_number += 1
            elif(current_val_number < val_model_number):
                val_ids.append(model_id)
                current_val_number += 1
            else:
                break

        print(f"{model_cat}: train -> {model_num - current_test_number - current_val_number}, val -> {current_val_number}, test -> {current_test_number}")

    test_ids_file = open(TESTIDSPATH, 'w')
    json.dump(test_ids, test_ids_file)
    test_ids_file.close()

    val_ids_file = open(VALPATH, 'w')
    json.dump(val_ids, val_ids_file)
    val_ids_file.close()