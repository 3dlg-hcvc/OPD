import glob
import os
import re
from time import time

TRAINOUTPUT = "/scratch/hanxiao/proj-motionnet/train_output/done"
OUTPUTBASE = "/scratch/hanxiao/proj-motionnet"
OUTPUTNAME = "extract"

experiment_names = ['motion_finetuning_cc_0.001_60000_0.5', 'motion_finetuning_oc_0.001_60000_0.5_15', 'motion_finetuning_ocv0_ew_dei_0.001_60000_36000_48000_0.5_15_0']

inference_file = True

def existDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def sorted_alphanum(file_list):
    """sort the file list by arrange the numbers in filenames in increasing order
    Designed for below case: img1, img2, img10
    If without this function: the sorted will be img1, img10, img2 (based on ASCII)
    With this function, the result will be img1, img2, img10 (Will split the number from the file name)
    :param file_list: a file list
    :return: sorted file list
    """
    if len(file_list) <= 1:
        return file_list
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(file_list, key=alphanum_key)

if __name__ == "__main__":
    start = time()

    existDir(f'{OUTPUTBASE}/{OUTPUTNAME}')
    
    for dir in experiment_names:
        existDir(f'{OUTPUTBASE}/{OUTPUTNAME}/{dir}')
        experiments = os.listdir(f'{TRAINOUTPUT}/{dir}')
        for experiment in experiments:
            models = glob.glob(f'{TRAINOUTPUT}/{dir}/{experiment}/model_final.pth')
            # If there is no model, it means that this experiment fails
            if len(models) == 0:
                continue
            # Even if the experiment fails, it's still valuable to save the log
            existDir(f'{OUTPUTBASE}/{OUTPUTNAME}/{dir}/{experiment}')

            if inference_file == True:
                existDir(f'{OUTPUTBASE}/{OUTPUTNAME}/{dir}/{experiment}/inference')
                # Copy the inference file
                os.system(f'cp {TRAINOUTPUT}/{dir}/{experiment}/inference/instances_predictions.pth {OUTPUTBASE}/{OUTPUTNAME}/{dir}/{experiment}/inference/instances_predictions.pth')
            else:
                # Copy the model_final file
                os.system(f'cp {TRAINOUTPUT}/{dir}/{experiment}/model_final.pth {OUTPUTBASE}/{OUTPUTNAME}/{dir}/{experiment}/model_final.pth')

    os.system(f'cd {OUTPUTBASE} && zip -r {OUTPUTNAME}.zip {OUTPUTNAME}')
    # os.system(f'rm -rf {OUTPUTBASE}/{OUTPUTNAME}')

    stop = time()
    print("Total Time: " + str(stop - start) + " seconds")