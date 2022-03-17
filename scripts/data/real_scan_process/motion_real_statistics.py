import os
import glob
import json
import numpy as np

data_dir = {"all": "/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/output/output"}

name_map_path = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/real_scan_process/real_name.json"

# TESTIDSPATH = '/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/raw_data_process/preprocess/testIds_real.json'
# VALIDIDPATH = '/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/raw_data_process/preprocess/validIds_real.json'

# datasets = ['train', 'val', 'test']
datasets = ['all']

if __name__ == "__main__":

    with open(name_map_path) as f:
        name_map = json.load(f)

    total_stat = {}

    for dataset in datasets:

        current_object = {}
        current_object_number = 0
        current_scan_number = 0
        current_image_number = 0
        dirs = glob.glob(f"{data_dir[dataset]}/*")
        for dir in dirs:
            # print(f"working on {dir}")
            model_name = dir.split('/')[-1]
            current_object[model_name] = glob.glob(f"{dir}/origin/*")
            current_object_number += 1
            current_image_number += len(current_object[model_name])
            scan_ids = []
            problem_scan_ids = []
            for image in current_object[model_name]:
                image_name = (image.split('/')[-1]).split('.')[0]
                scan_id = (image.split('/')[-1]).split('.')[0][:50]
                if not os.path.isfile(f"{dir}/origin_annotation/{image_name}.json"):
                    print(f"No annotation: {image}")
                    import pdb
                    pdb.set_trace()
                    # os.system(f"rm -rf {image}")
                    # os.system(f"rm -rf {dir}/depth/{image_name}_d.png")
                    # os.system(f"rm -rf {dir}/mask/{image_name}_*.png")
                    # if scan_id not in problem_scan_ids:
                    #     problem_scan_ids.append(scan_id)
                    # else:
                    #     print("Multiple images for the problem scan")
                    # continue
                if scan_id not in scan_ids:
                    current_scan_number += 1
                    scan_ids.append(scan_id)
                # Read the motion number
                with open(f"{dir}/origin_annotation/{image_name}.json") as f:
                    anno = json.load(f)
                # Make it consistent with 2DMotion dataset
                extrinsic_matrix = np.linalg.inv(np.reshape(anno["camera"]["extrinsic"]["matrix"], (4, 4), order="F")).flatten(order="F")
                anno["camera"]["extrinsic"]["matrix"] = list(extrinsic_matrix)
                with open(f"{dir}/origin_annotation/{image_name}.json", 'w') as f:
                    json.dump(anno, f)
                
                motion_number = len(anno["motions"])
                motion_ids = [anno["partId"] for anno in anno["motions"]]
                mask_paths = glob.glob(f"{dir}/mask/{image_name}_*")
                if not motion_number == len(mask_paths):
                    print(f"Not consistent mask and motion {image}")
                # Rename the RGB
                model_name = image_name.rsplit('-', 1)[0]
                new_image_name = name_map[model_name] + '-' + image_name.rsplit('-', 1)[1]
                os.system(f"mv {dir}/origin/{image_name}.png {dir}/origin/{new_image_name}.png")
                # Rename the depth
                os.system(f"mv {dir}/depth/{image_name}_d.png {dir}/depth/{new_image_name}_d.png")
                # Rename the annotation
                os.system(f"mv {dir}/origin_annotation/{image_name}.json {dir}/origin_annotation/{new_image_name}.json")
                # Rename all the masks
                for mask_path in mask_paths:
                    mask_name = (mask_path.split('/')[-1]).split('.')[0]
                    if mask_name.rsplit('_', 1)[1] not in motion_ids:
                        import pdb
                        pdb.set_trace()
                    new_mask_name = f"{new_image_name}_{mask_name.rsplit('_', 1)[1]}"
                    os.system(f"mv {dir}/mask/{mask_name}.png {dir}/mask/{new_mask_name}.png")
        total_stat[dataset] = current_object

        print(f"{dataset} Set -> Object Number {current_object_number}, Scan Number {current_scan_number}, Image Number {current_image_number}, Avg Images Per Object {current_image_number/current_object_number}, Avg Images Per Scan {current_image_number/current_scan_number}")
    
    # val_ids_file = open(VALIDIDPATH, 'w')
    # json.dump(list(total_stat["val"].keys()), val_ids_file)
    # val_ids_file.close()
    
    # test_ids_file = open(TESTIDSPATH, 'w')
    # json.dump(list(total_stat["test"].keys()), test_ids_file)
    # test_ids_file.close()
