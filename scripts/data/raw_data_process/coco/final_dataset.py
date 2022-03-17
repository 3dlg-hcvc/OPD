import glob
import os

PROCESSPATH = '/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/process_data_real/'
DATASETPATH = '/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/MotionDataset_real/'

def existDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

if __name__ == "__main__":
    # Create the dirs
    dir_names = ['train/', 'valid/', 'test/', 'annotations/', 'depth/']
    for dir_name in dir_names:
        existDir(DATASETPATH + dir_name)

    # Move the origin images and depth images
    origin_dir = ['train/', 'valid/', 'test/']
    for dir_name in origin_dir:
        print(f'Copying the {dir_name} images')

        # Move the origin images
        input_path = f'{PROCESSPATH}{dir_name}origin/'
        output_path = f'{DATASETPATH}{dir_name}'
        # Loop the images
        file_paths = glob.glob(f'{input_path}*')
        for file_path in file_paths:
            file_name = file_path.split('/')[-1]
            new_file_path = f'{output_path}{file_name}'
            os.system(f'cp {file_path} {new_file_path}')

        # Move the depth images
        input_path = f'{PROCESSPATH}{dir_name}depth/'
        output_path = f'{DATASETPATH}depth/'
        # Loop the images
        file_paths = glob.glob(f'{input_path}*')
        for file_path in file_paths:
            file_name = file_path.split('/')[-1]
            new_file_path = f'{output_path}{file_name}'
            os.system(f'cp {file_path} {new_file_path}')

    # Move the annotations
    file_paths = glob.glob(f'{PROCESSPATH}coco_annotation/*')
    for file_path in file_paths:
        print('Copying the coco annotations')
        file_name = file_path.split('/')[-1]
        new_file_path = f'{DATASETPATH}annotations/{file_name}'
        os.system(f'cp {file_path} {new_file_path}')

