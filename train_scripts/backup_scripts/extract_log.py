import glob
import os
import re
from time import time

TRAINOUTPUT = "/scratch/hanxiao/proj-motionnet/train_output"
OUTPUTBASE = "/scratch/hanxiao/proj-motionnet"
OUTPUTNAME = "extract"

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
    
    existDir(f'{TRAINOUTPUT}/done')
    dirs = os.listdir(TRAINOUTPUT)
    for dir in dirs:
        # Ignore the dir done (The experiments which has been extracted log)
        if dir == "done":
            continue
        existDir(f'{OUTPUTBASE}/{OUTPUTNAME}/{dir}')
        experiments = os.listdir(f'{TRAINOUTPUT}/{dir}')
        for experiment in experiments:
            # models = glob.glob(f'{TRAINOUTPUT}/{dir}/{experiment}/*.pth')
            # # If there is no model, it means that this experiment fails
            # if len(models) == 0:
            #     continue
            # Even if the experiment fails, it's still valuable to save the log
            existDir(f'{OUTPUTBASE}/{OUTPUTNAME}/{dir}/{experiment}')
            # Only copy the latest log, other logs are related to resubmitting jobs
            logfiles = sorted_alphanum(glob.glob(f'{TRAINOUTPUT}/{dir}/{experiment}/*.out'))
            # if len(logfiles) > 1:
            #     raise ValueError(f"The number of log is not correct for {TRAINOUTPUT}/{dir}/{experiment}") 
            # Copy the log file
            os.system(f'cp {logfiles[-1]} {OUTPUTBASE}/{OUTPUTNAME}/{dir}/{experiment}/{logfiles[-1].split("/")[-1]}')
            tbfiles = glob.glob(f'{TRAINOUTPUT}/{dir}/{experiment}/events*')
            if len(tbfiles) > 1:
                raise ValueError(f"The number of tensorboard is not correct for {TRAINOUTPUT}/{dir}/{experiment}") 
            # Copy the tensorboard file
            if len(tbfiles) > 0:
                os.system(f'cp {tbfiles[0]} {OUTPUTBASE}/{OUTPUTNAME}/{dir}/{experiment}/{tbfiles[0].split("/")[-1]}')
            # Copy the inference dirs
            if os.path.exists(f'{TRAINOUTPUT}/{dir}/{experiment}/inference'):
                existDir(f'{OUTPUTBASE}/{OUTPUTNAME}/{dir}/{experiment}/inference')
                os.system(f'cp {TRAINOUTPUT}/{dir}/{experiment}/inference/*.pth {OUTPUTBASE}/{OUTPUTNAME}/{dir}/{experiment}/inference/')

    os.system(f'cd {OUTPUTBASE} && zip -r {OUTPUTNAME}.zip {OUTPUTNAME}')
    os.system(f'rm -rf {OUTPUTBASE}/{OUTPUTNAME}')

    stop = time()
    print("Total Time: " + str(stop - start) + " seconds")