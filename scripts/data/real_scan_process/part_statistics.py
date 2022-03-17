import csv
import json
import requests

annotation_path = "/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/real_scan_process/annotation.csv"
TESTIDSPATH = '/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/raw_data_process/preprocess/testIds_real.json'
VALIDIDPATH = '/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/raw_data_process/preprocess/validIds_real.json'

def get_web_scans_list(scans_list_url):
    res = requests.get(scans_list_url)
    scans_list = res.json().get('data', None)
    return scans_list

if __name__ == '__main__':
    file = open(annotation_path, "r")
    csv_reader = csv.reader(file)
    index = 0
    models = {}
    for line in csv_reader:
        if index == 0:
            index += 1
            continue
        model_index = line[4]
        model_cat = line[7]
        model_quality = line[8]
        scan_id = line[5]

        if model_index not in models.keys():
            models[model_index] = [(model_cat, model_quality, scan_id)]
        else:
            models[model_index].append((model_cat, model_quality, scan_id))
        index += 1

    # Check if the same object has the same obj cat for different scans
    for model_ind in models.keys():
        cat = models[model_ind][0][0]
        for i in range(1, len(models[model_ind])):
            if not models[model_ind][i][0] == cat:
                print(f"ERROR: {model_ind}")
    
    # Get the number of valid objects
    num_valid_obj = 0
    num_valid_scan = 0
    valid_objs = []
    for model_ind in models.keys():
        valid = False
        for i in range(len(models[model_ind])):
            if not (models[model_ind][i][1]).lower() == "bad":
                valid = True
                num_valid_scan += 1
        if valid == True:
            num_valid_obj += 1
            valid_objs.append(model_ind)
    print(f"After annotating, there are {num_valid_obj} valid objects with {num_valid_scan} scans")

    # # Statistics on the valid objs 
    # model_cats = {}
    # for model_ind in models.keys():
    #     if model_ind not in valid_objs:
    #         continue
    #     model_cat = models[model_ind][0][0]
    #     if model_cat not in model_cats.keys():
    #         model_cats[model_cat] = 1
    #     else:
    #         model_cats[model_cat] += 1

    # print('\n')
    # print("Model category statistics")
    # print(model_cats)

    test_ids_file = open(TESTIDSPATH)
    test_ids = json.load(test_ids_file)
    test_ids_file.close()

    valid_ids_file = open(VALIDIDPATH)
    valid_ids = json.load(valid_ids_file)
    valid_ids_file.close()



    part_labels = {"train": {"drawer": 0, "door": 0, "lid": 0, "translation": 0, "rotation": 0}, "val": {"drawer": 0, "door": 0, "lid": 0, "translation": 0, "rotation": 0}, "test": {"drawer": 0, "door": 0, "lid": 0, "translation": 0, "rotation": 0}}
    part_cats = {}
    for model_ind in models.keys():
        if model_ind not in valid_objs:
            continue

        if model_ind in valid_ids:
            set_name = "val"
        elif model_ind in test_ids:
            set_name = "test"
        else:
            set_name = "train"
        
        part_anno = get_web_scans_list(f"https://aspis.cmpt.sfu.ca/stk-multiscan/articulation-annotations/load-annotations?modelId=multiscan.{models[model_ind][0][2]}")["parts"]

        part_cat = models[model_ind][0][0].strip()
        if part_cat not in part_cats.keys():
            part_cats[part_cat] = 0

        for part in part_anno:
            part_name = None
            if "drawer" in part["label"]:
                part_name = "drawer"
            elif "door" in part["label"]:
                part_name = "door"
            elif "lid" in part["label"]:
                part_name = "lid"
            if part_name != None:
                part_cats[part_cat] += 1
                part_labels[set_name][part_name] += 1
        
    print(part_labels)
    print(part_cats)

