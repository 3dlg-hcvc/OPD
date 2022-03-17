# Statistics on the min_bound and max_bound

import json
import glob
import numpy as np

object_info_path = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/output_object_info"

if __name__ == "__main__":
    datasets = ["."]

    diagonal_dict = {}
    scan_index = 0
    real_name_map = {}
    for dataset in datasets:
        anno_paths = glob.glob(f"{object_info_path}/{dataset}/*.json")
        for anno_path in anno_paths:
            anno_file = open(anno_path)
            anno = json.load(anno_file)
            anno_file.close()

            if not len(anno.keys()) == 1:
                print(f"Something wrong: {anno_path}")
            if list(anno.keys())[0] in diagonal_dict.keys():
                print(f"Something wrong 2: {anno_path} {diagonal_dict[list(anno.keys())[0]]}")
            
            diameter = anno[list(anno.keys())[0]]["diameter"]
            min_bound = np.array(anno[list(anno.keys())[0]]["min_bound"])
            max_bound = np.array(anno[list(anno.keys())[0]]["max_bound"])

            diagonal_length = ((max_bound - min_bound) ** 2).sum()**0.5

            import pdb
            pdb.set_trace()
    
   