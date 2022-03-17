import os
import csv


MODEL_INFO_PATH = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/ancsh_data/split_info.csv"
RAWDATAPATH = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/raw_data_6.1"
NEWDATAPATH = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/raw_data_ancsh"


def existDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

if __name__ == "__main__":
    existDir(NEWDATAPATH)

    f = open(MODEL_INFO_PATH, "r")
    csv_reader = csv.reader(f)

    models = {"train": [], "val": [], "test": []}
    index = 0
    for line in csv_reader:
        if index == 0:
            index += 1
            continue
        
        dataset = line[0]
        models[dataset].append(line[3])

        index += 1

    all_models = models["train"] + models["val"] + models["test"]

    for model in all_models:
        os.system(f"cp -r {RAWDATAPATH}/{model} {NEWDATAPATH}/{model}")