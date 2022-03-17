import glob
import json
import numpy as np
from time import time

RAWDATAPATH = "/home/hja40/Desktop/Research/proj-motionnet/Dataset/raw_data_6.0/"

if __name__ == "__main__":
    start = time()
    model_paths = glob.glob(RAWDATAPATH + "*")

    dimension_num = 0
    dimension_mean = np.zeros(3)

    for model_path in model_paths:
        # Accumulate the dimension 
        json_paths = glob.glob(model_path + "/origin_annotation/*.json")
        for json_path in json_paths:
            json_file = open(json_path)
            annotation = json.load(json_file)
            json_file.close()
            motions = annotation["motions"]
            for motion in motions:
                dimension_num += 1
                dimension_mean += motion['partPose']['dimension']

    dimension_mean /= dimension_num
    print(f'Mean Dimension: {dimension_mean}')
    
    stop = time()
    print(str(stop - start) + " seconds")