import glob
import os
import re
from time import time

TRAINOUTPUT = "/scratch/hanxiao/proj-motionnet/train_output"
OUTPUTBASE = "/scratch/hanxiao/proj-motionnet"
OUTPUTNAME = "log.txt"

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

    output = []

    dirs = os.listdir(TRAINOUTPUT)
    for dir in dirs:
        if dir == "done":
            continue
        output.append(f"{dir}:")
        experiments = sorted_alphanum(os.listdir(f'{TRAINOUTPUT}/{dir}'))
        for experiment in experiments:
            logfiles = sorted_alphanum(glob.glob(f'{TRAINOUTPUT}/{dir}/{experiment}/*.out'))
            # Because of cedar uncleaned GPU problem, resubmitting jobs may cause multiple log files
            # We just want to check the latest log (with the biggest job id)
            logfile = open(logfiles[-1], "r")
            lines = logfile.readlines()
            logfile.close()
            output.append(f"    {experiment}: {lines[len(lines)-1]}")
        output.append("")

    file = open(f"{OUTPUTBASE}/{OUTPUTNAME}", "w+")
    file.write("\n".join(output))

    stop = time()
    print("Total Time: " + str(stop - start) + " seconds")