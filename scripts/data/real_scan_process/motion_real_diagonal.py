# Run this code, then statistics to process the MotionREAL data

import json
import glob

object_info_path = "/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/output/output_object_info"
real_attr_path = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/data_statistics/real-attr.json"
real_name_map_path = "./real_name.json"

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
            diagonal_dict[str(scan_index)] = {"diameter": anno[list(anno.keys())[0]]["diameter"], "min_bound": anno[list(anno.keys())[0]]["min_bound"], "max_bound": anno[list(anno.keys())[0]]["max_bound"]}
            # diagonal_dict[list(anno.keys())[0]] = anno_path
            real_name_map[list(anno.keys())[0].rsplit('_', 1)[0] + '_1'] = str(scan_index)
            scan_index += 1
    
    diagonal_file = open(real_attr_path, 'w')
    json.dump(diagonal_dict, diagonal_file)
    diagonal_file.close()

    map_file = open(real_name_map_path, 'w')
    json.dump(real_name_map, map_file)
    map_file.close()