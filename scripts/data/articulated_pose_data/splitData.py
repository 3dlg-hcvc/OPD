import glob
import os
import json
import multiprocessing
from time import time

# Split the raw dataset into train/valid/test set based on the splitted model ids
TRAINPATH = '/cs/3dlg-project/3dlg-hcvc/motionnet/Dataset/splits/drawer/0.01/train.txt'
TESTPATH = '/cs/3dlg-project/3dlg-hcvc/motionnet/Dataset/splits/drawer/0.01/test.txt'
IMAGEMAP = '/local-scratch/localhome/hja40/Desktop/Dataset/raw_data_articulated_pose/image_map.json'
RAWDATAPATH = '/local-scratch/localhome/hja40/Desktop/Dataset/raw_data_6.2/'
OUTPUTDATAPATH = '/local-scratch/localhome/hja40/Desktop/Dataset/process_data_6.2/'
unseen_model = [ '46123',  '45841', '46440']

def existDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def processImage(image_path, train_test):
    dir_names = ['origin', 'mask', 'depth', 'origin_annotation']
    tokens = image_path.split('-')
    model_id = tokens[0]
    
    
    for dir_name in dir_names:
        # print(f'{RAWDATAPATH}{model_id}/{dir_name}/{image_path}*')
        if dir_name == 'mask':
            file_paths = glob.glob(f'{RAWDATAPATH}{model_id}/{dir_name}/{image_path}_*')
        elif dir_name == 'origin':
            file_paths = [f'{RAWDATAPATH}{model_id}/{dir_name}/{image_path}.png']
        elif dir_name == 'depth':
            file_paths = [f'{RAWDATAPATH}{model_id}/{dir_name}/{image_path}_d.png']
        else:
            file_paths = [f'{RAWDATAPATH}{model_id}/{dir_name}/{image_path}.json']
        for file_path in file_paths:
            file_name = file_path.split('/')[-1]
            process_path = OUTPUTDATAPATH + train_test + '/' + dir_name + '/' + file_name
            os.system(f'cp {file_path} {process_path}')

if __name__ == "__main__":
    start = time()
    pool = multiprocessing.Pool(processes=16)

    json_file = open(IMAGEMAP)
    image_map = json.load(json_file)
    json_file.close()


    train_list = []
    valid_list = []
    test_list = []

    # Process the train images
    train_file = open(TRAINPATH)
    train_paths = train_file.readlines()   
    train_paths = [x.split('.h5')[0] for x in train_paths]
    for train_path in train_paths:
        tokens = train_path.split('/')
        viewId = tokens[-1]
        poseId = tokens[-2]
        modelId = tokens[-3]
        # print(image_map[f'{modelId}_{poseId}_{viewId}'])
        train_list.append(image_map[f'{modelId}_{poseId}_{viewId}'])

    # Process the test and valid images
    raw_file = open(TESTPATH)
    raw_paths = raw_file.readlines()
    raw_paths = [x.split('.h5')[0] for x in raw_paths]
    for raw_path in raw_paths:
        tokens = raw_path.split('/')
        viewId = tokens[-1]
        poseId = tokens[-2]
        modelId = tokens[-3]
        if modelId in unseen_model:
            test_list.append(image_map[f'{modelId}_{poseId}_{viewId}'])
        else:
            valid_list.append(image_map[f'{modelId}_{poseId}_{viewId}'])
        

    dir_names = ['origin/', 'mask/', 'depth/', 'origin_annotation/']

    for dir_name in dir_names:
        existDir(OUTPUTDATAPATH + 'train/' + dir_name)
        existDir(OUTPUTDATAPATH + 'valid/' + dir_name)
        existDir(OUTPUTDATAPATH + 'test/' + dir_name)

    # Copy the train images RGB, depth, mask and corresponding annotation
    for image in train_list:
        # processImage(image, 'train')
        pool.apply_async(processImage, (image, 'train',))
    
    for image in valid_list:
        # processImage(image, 'valid')
        pool.apply_async(processImage, (image, 'valid',))

    for image in test_list:
        # processImage(image, 'test')
        pool.apply_async(processImage, (image, 'test',))


    pool.close()
    pool.join()

    stop = time()
    print(str(stop - start) + " seconds")

