import json
import glob
from PIL import Image
import numpy as np
import csv
import requests

annotation_path = "/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/real_scan_process/annotation.csv"
PART_CAT = ["drawer", "door", "lid"]
MODEL_CAT = ['StorageFurniture', 'Table', 'TrashCan', 'Refrigerator', 'Microwave', 'WashingMachine', 'Dishwasher', 'Oven', 'Safe', 'Box', 'Suitcase'] 
structure_statistics = {}

detailed_info = []

def get_web_scans_list(scans_list_url):
    res = requests.get(scans_list_url)
    scans_list = res.json().get('data', None)
    return scans_list

def processModel(model):
    part_anno = get_web_scans_list(f"https://aspis.cmpt.sfu.ca/stk-multiscan/articulation-annotations/load-annotations?modelId=multiscan.{model[0][2]}")["parts"]
    model_cat = model[0][0].strip()
    model_id = model[0][2]

    for part in part_anno:
        part_cat = None
        if "drawer" in part["label"]:
            part_cat = "drawer"
        elif "door" in part["label"]:
            part_cat = "door"
        elif "lid" in part["label"]:
            part_cat = "lid"
        if part_cat != None:
            part_id = part["pid"]
            detailed_info.append([model_id, model_cat, part_id, part_cat])
    # import pdb
    # pdb.set_trace()
    # result = {"drawer": 0, "door": 0, "lid": 0, "translation": 0, "rotation": 0}
    # annotation_paths = glob.glob(f"{model}/origin_annotation/*")

    # annotation_path = annotation_paths[0]
    # anno_file = open(annotation_path)
    # annotation = json.load(anno_file)
    # anno_file.close()

    # model_id = model.split('/')[-1]
    # model_cat = annotation["label"].strip()
    # if model_cat not in MODEL_CAT:
    #     raise ValueError(f"Not valid model category {model_cat}")

    # motions = annotation["motions"]
    # # The order for structure is drawer, door and lid
    # structure = [0, 0, 0]
    # for motion in motions:
    #     part_cat = motion["label"].strip()
    #     part_id = motion["partId"]
    #     if part_cat not in PART_CAT:
    #         raise ValueError(f"Not valid part category {part_cat}")
    #     structure[PART_CAT.index(part_cat)] += 1
    #     detailed_info.append([model_id, model_cat, part_id, part_cat])

    # structure = [str(x) for x in structure]
    # structure_name = "_".join(structure)

    # if structure_name not in structure_statistics:
    #     structure_statistics[structure_name] = {}
    #     for cat in MODEL_CAT:
    #         structure_statistics[structure_name][cat] = 0
    # structure_statistics[structure_name][model_cat] += 1
    

if __name__ == "__main__":
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

    for model_ind in models.keys():
        if model_ind not in valid_objs:
            continue
        processModel(models[model_ind])

    # output_path = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/real_scan_process/structure_statistics.csv"
    # file = open(output_path, "w")
    # csv_write = csv.writer(file)
    # csv_head = ["structure"] + MODEL_CAT
    # csv_write.writerow(csv_head)

    # for k, v in structure_statistics.items():
    #     data = []
    #     data.append(k)
    #     for cat in MODEL_CAT:
    #         data.append(v[cat])
    #     csv_write.writerow(data)

    # file.close()

    output_path = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/real_scan_process/structure_statistics_detailed.csv"
    file = open(output_path, "w")
    csv_write = csv.writer(file)
    csv_head = ["dataset", "object_id", "object_category", "part_id", "part_category"]
    csv_write.writerow(csv_head)

    for k in detailed_info:
        csv_write.writerow(["real"] + k)

    file.close()
    
