import glob
import os
import json
import multiprocessing
from time import time

# Split the raw dataset into train/valid/test set based on the splitted model ids
TESTIDSPATH = '/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/raw_data_process/preprocess/testIds_real.json'
VALIDIDPATH = '/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/raw_data_process/preprocess/validIds_real.json'
RAWDATAPATH = '/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/raw_data_real/'
OUTPUTDATAPATH = '/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/process_data_real/'

def existDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def process(model_path, train_test):
    dir_names = ['origin', 'mask', 'depth', 'origin_annotation']
    
    for dir_name in dir_names:
        file_paths = glob.glob(f'{model_path}/{dir_name}/*')
        for file_path in file_paths:
            file_name = file_path.split('/')[-1]
            process_path = OUTPUTDATAPATH + train_test + '/' + dir_name + '/' + file_name
            os.system(f'cp {file_path} {process_path}')

if __name__ == "__main__":
    start = time()
    pool = multiprocessing.Pool(processes=16)

    test_ids_file = open(TESTIDSPATH)
    test_ids = json.load(test_ids_file)
    test_ids_file.close()

    valid_ids_file = open(VALIDIDPATH)
    valid_ids = json.load(valid_ids_file)
    valid_ids_file.close()

    dir_names = ['origin/', 'mask/', 'depth/', 'origin_annotation/']

    for dir_name in dir_names:
        existDir(OUTPUTDATAPATH + 'train/' + dir_name)
        existDir(OUTPUTDATAPATH + 'valid/' + dir_name)
        existDir(OUTPUTDATAPATH + 'test/' + dir_name)

    model_paths = glob.glob(RAWDATAPATH + '*')
    for model_path in model_paths:
        model_id = model_path.split('/')[-1]
       
        if model_id in test_ids:
            # process(model_path, 'test')
            pool.apply_async(process, (model_path, 'test',))
        elif model_id in valid_ids:
            # process(model_path, 'valid')
            pool.apply_async(process, (model_path, 'valid',))
        else:
            # process(model_path, 'train')
            pool.apply_async(process, (model_path, 'train',))

    pool.close()
    pool.join()

    stop = time()
    print(str(stop - start) + " seconds")

