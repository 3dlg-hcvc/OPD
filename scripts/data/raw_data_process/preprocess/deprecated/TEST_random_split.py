import glob
import os
import random

RAWDATAPATH = '../../../raw_data_2.0/'
OUTPUTDATAPATH = '../../../TEST_process_data/'

def existDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

if __name__ == "__main__":
    dir_names = ['origin/', 'mask/', 'depth/', 'origin_annotation/']
    for dir_name in dir_names:
        existDir(OUTPUTDATAPATH + 'train/' + dir_name)
        existDir(OUTPUTDATAPATH + 'valid/' + dir_name)
        existDir(OUTPUTDATAPATH + 'test/' + dir_name)
    
    model_paths = glob.glob(RAWDATAPATH + '*')
    for model_path in model_paths:
        print(model_path)
        file_paths = glob.glob(f'{model_path}/origin/*')
        for file_path in file_paths:
            img_name = (file_path.split('/')[-1]).split('.')[0]
            rand = random.random()
            if(rand < 0.7):
                train_test = 'train'
            elif(rand < 0.95):
                train_test = 'test'
            else:
                train_test = 'valid'
            # Copy the origin image
            origin_file_path = f'{model_path}/origin/{img_name}.png'
            new_origin_path = f'{OUTPUTDATAPATH}{train_test}/origin/{img_name}.png'
            os.system(f'cp {origin_file_path} {new_origin_path}')

            # Copy the depth image (One origin <-> One depth)
            depth_file_path = f'{model_path}/depth/{img_name}_d.png'
            new_depth_path = f'{OUTPUTDATAPATH}{train_test}/depth/{img_name}_d.png'
            os.system(f'cp {depth_file_path} {new_depth_path}')

            # Copy all mask images (One origin <-> Multiple masks)
            mask_file_paths = glob.glob(f'{model_path}/mask/{img_name}_*')
            for mask_file_path in mask_file_paths:
                mask_file_name = mask_file_path.split('/')[-1]
                new_mask_path = f'{OUTPUTDATAPATH}{train_test}/mask/{mask_file_name}'
                os.system(f'cp {mask_file_path} {new_mask_path}')

            # Copy the annotation (One origin <-> One annotation)
            anno_file_path = f'{model_path}/origin_annotation/{img_name}.json'
            new_anno_path = f'{OUTPUTDATAPATH}{train_test}/origin_annotation/{img_name}.json'
            os.system(f'cp {anno_file_path} {new_anno_path}')