import os
import json
from time import time
import argparse
import re

VALIDLISTFILE = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/vis_scripts/valid_all.json"
OUTPUTDIR = "/project/3dlg-hcvc/motionnet/www/eccv-opdet-synthetic/render"
RAWDATAPATH = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/MotionDataset_6.11"

# Some parameters for the render_pred.py
SCORETHRESHOLD = 0.8
DATAPATH = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/MotionDataset_6.11"
# INFERENCEBASEPATH = "/project/3dlg-hcvc/motionnet/experiments/finetuning"
INFERENCEBASEPATH = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/experiments/finetuning"
# if True, update both annotation image and annotation file for the prediction, or only annotation file
update_all = True
render_gt = True
print_command = False

def sorted_alphanum(file_list):
    """sort the file list by arrange the numbers in filenames in increasing order
    :param file_list: a file list
    :return: sorted file list
    """
    if len(file_list) <= 1:
        return file_list
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(file_list, key=alphanum_key)

def get_folder_list(path):
    if not os.path.exists(path):
        raise OSError('Path {} not exist!'.format(path))

    dir_list = []
    for dirname in os.listdir(path):
        if os.path.isdir(os.path.join(path, dirname)):
            dir_list.append(os.path.join(path, dirname))
    dir_list = sorted_alphanum(dir_list)
    return dir_list

if __name__ == "__main__":
    start = time()

    parser = argparse.ArgumentParser(description='Motionnet Visualization')
    parser.add_argument('-ms', '--models', dest='models', action='store',
                        nargs="+",
                        default=[
                            "ocv0_rgbd", "cc_rgbd", "mf_rgbd", "rm_rgbd", "ancsh_rgb", "pc_rgb"
                        ],
                        help='models')
    args = parser.parse_args()

    # Move the picked valid and depth image into the render/valid and render/depth
    valid_image_file = open(VALIDLISTFILE)
    selection = json.load(valid_image_file)
    valid_image_file.close()

    os.makedirs(f'{OUTPUTDIR}/valid', exist_ok=True)
    os.makedirs(f'{OUTPUTDIR}/depth', exist_ok=True)

    if update_all and render_gt:
        for image_name in selection:
            # Copy the image into render/valid
            os.system(f'cp {RAWDATAPATH}/valid/{image_name}.png {OUTPUTDIR}/valid/')
            os.system(f'cp {RAWDATAPATH}/depth/{image_name}_d.png {OUTPUTDIR}/depth/')

        # gt
        if not print_command:
            os.system(
                f"python render_gt.py --valid-image {VALIDLISTFILE} --output-dir {OUTPUTDIR}/gt --data-path {DATAPATH} &"
            )
        else:
            print(f"python render_gt.py --valid-image {VALIDLISTFILE} --output-dir {OUTPUTDIR}/gt --data-path {DATAPATH} &")
    
    if update_all == True:
        update = '--update-all'
    else:
        update = ''

    for model_name in args.models:
        eval_dirs = get_folder_list(f'{INFERENCEBASEPATH}/{model_name}/eval_output')
        if len(eval_dirs) == 0:
            raise ValueError(f'No inference file for model {model_name}')
        eval_dir = eval_dirs[-1]
        inference_file = f'{eval_dir}/inference/instances_predictions.pth'
        input_type = model_name.split('_')[-1]
        model_type = model_name.split('_')[0]
        if model_type == "ancsh" or model_type == "pc":
            extra = "--no_mask"
        else:
            extra = ""
        if input_type == "depth":
            if not print_command:
                os.system(
                    f"python render_pred.py --score-threshold {SCORETHRESHOLD} {update} --valid-image {VALIDLISTFILE} --depth --output-dir {OUTPUTDIR}/{model_name} --data-path {DATAPATH} --inference-file {inference_file} {extra} &"
                )
            else:
                print(f"python render_pred.py --score-threshold {SCORETHRESHOLD} {update} --valid-image {VALIDLISTFILE} --depth --output-dir {OUTPUTDIR}/{model_name} --data-path {DATAPATH} --inference-file {inference_file} {extra} &")
        else:
            if not print_command:
                os.system(
                    f"python render_pred.py --score-threshold {SCORETHRESHOLD} {update} --valid-image {VALIDLISTFILE} --output-dir {OUTPUTDIR}/{model_name} --data-path {DATAPATH} --inference-file {inference_file} {extra} &"
                )
            else:
                print(f"python render_pred.py --score-threshold {SCORETHRESHOLD} {update} --valid-image {VALIDLISTFILE} --output-dir {OUTPUTDIR}/{model_name} --data-path {DATAPATH} --inference-file {inference_file} {extra} &")

    stop = time()
    print(str(stop - start) + " seconds")
