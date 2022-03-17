import json
import glob
from PIL import Image
import numpy as np
import csv

RAWDATAPATH = "/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/raw_data_6.1/"
PART_CAT = ["drawer", "door", "lid"]
MODEL_CAT = ['StorageFurniture', 'Table', 'TrashCan', 'Refrigerator', 'Microwave', 'WashingMachine', 'Dishwasher', 'Oven', 'Safe', 'Box', 'Suitcase'] 
structure_statistics = {}

detailed_info = []

def processModel(model):
    result = {"drawer": 0, "door": 0, "lid": 0, "translation": 0, "rotation": 0}
    annotation_paths = glob.glob(f"{model}/origin_annotation/*")

    annotation_path = annotation_paths[0]
    anno_file = open(annotation_path)
    annotation = json.load(anno_file)
    anno_file.close()

    model_id = model.split('/')[-1]
    model_cat = annotation["label"].strip()
    if model_cat not in MODEL_CAT:
        raise ValueError(f"Not valid model category {model_cat}")

    motions = annotation["motions"]
    # The order for structure is drawer, door and lid
    structure = [0, 0, 0]
    for motion in motions:
        part_cat = motion["label"].strip()
        part_id = motion["partId"]
        if part_cat not in PART_CAT:
            raise ValueError(f"Not valid part category {part_cat}")
        structure[PART_CAT.index(part_cat)] += 1
        detailed_info.append([model_id, model_cat, part_id, part_cat])

    structure = [str(x) for x in structure]
    structure_name = "_".join(structure)

    if structure_name not in structure_statistics:
        structure_statistics[structure_name] = {}
        for cat in MODEL_CAT:
            structure_statistics[structure_name][cat] = 0
    structure_statistics[structure_name][model_cat] += 1
    

if __name__ == "__main__":
    # test_ids_file = open(TESTIDSPATH)
    # test_ids = json.load(test_ids_file)
    # test_ids_file.close()

    # valid_ids_file = open(VALIDIDPATH)
    # valid_ids = json.load(valid_ids_file)
    # valid_ids_file.close()

    models = glob.glob(f"{RAWDATAPATH}/*")
    part_labels = {}
    for model in models:
        processModel(model)

    output_path = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/data_statistics/structure_statistics.csv"
    file = open(output_path, "w")
    csv_write = csv.writer(file)
    csv_head = ["structure"] + MODEL_CAT
    csv_write.writerow(csv_head)

    for k, v in structure_statistics.items():
        data = []
        data.append(k)
        for cat in MODEL_CAT:
            data.append(v[cat])
        csv_write.writerow(data)

    file.close()

    output_path = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/data_statistics/structure_statistics_detailed.csv"
    file = open(output_path, "w")
    csv_write = csv.writer(file)
    csv_head = ["dataset", "object_id", "object_category", "part_id", "part_category"]
    csv_write.writerow(csv_head)

    for k in detailed_info:
        csv_write.writerow(["synth"] + k)

    file.close()
    
