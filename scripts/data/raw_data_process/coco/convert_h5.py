import h5py
import os
import glob
import numpy as np
import multiprocessing
from time import time
from PIL import Image

DATASETPATH = '/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/MotionDataset_real'
NEWPATH = '/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/MotionDataset_h5_real'

def existDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def convert_h5(dir):
    global DATASETPATH, NEWPATH
    files = sorted(glob.glob(f'{DATASETPATH}/{dir}/*.png'))
    num_files = len(files)

    with h5py.File(f'{NEWPATH}/{dir}.h5', "a") as h5file:
        # Need to store the depth into a different format
        if dir == 'depth':
            first_img = img = np.asarray(Image.open(files[0]), dtype=np.float32)[:, :, None]
        else:
            first_img = img = np.asarray(Image.open(files[0]).convert("RGB"))
        img_shape = first_img.shape
        img_dtype = first_img.dtype
        dataset_shape = (num_files,) + img_shape
        chunk_shape = (1,) + img_shape
        string_dtype = h5py.string_dtype(encoding="utf-8")
        # create image dataset tensor
        dset_images = h5file.create_dataset(
            f"{dir}_images",
            shape=dataset_shape,
            dtype=img_dtype,
            chunks=chunk_shape,
            compression="gzip",
            compression_opts=9,
            shuffle=True,
        )
        # create image filenames dataset
        dset_filenames = h5file.create_dataset(
            f"{dir}_filenames", shape=(num_files,), dtype=string_dtype
        )
        # now fill the data
        for i in range(num_files):
            file = files[i]
            filepath = os.path.relpath(file, start=f'{DATASETPATH}/{dir}')
            if dir == 'depth':
                img = np.asarray(Image.open(file), dtype=np.float32)[:, :, None]
            else:
                img = np.asarray(Image.open(file).convert("RGB"))
            dset_images[i] = img
            dset_filenames[i] = filepath
            if i % 1000 == 0:
                print(f"{dir}: {i}/{num_files}")

if __name__ == "__main__":
    start = time()
    pool = multiprocessing.Pool(processes=16)

    # Create the annotations dir
    existDir(f'{NEWPATH}/annotations')
    # Move the annotations
    file_paths = glob.glob(f'{DATASETPATH}/annotations/*')
    for file_path in file_paths:
        print('Copying the coco annotations')
        file_name = file_path.split('/')[-1]
        new_file_path = f'{NEWPATH}/annotations/{file_name}'
        os.system(f'cp {file_path} {new_file_path}')

    # Convert the images in to h5 for train/test/valid/depth dir
    dirs = ['train', 'test', 'valid', 'depth']
    for dir in dirs:
        print(f'Processing {dir}')
        pool.apply_async(convert_h5, (dir,))
    
    pool.close()
    pool.join()

    stop = time()
    print(str(stop - start) + " seconds")