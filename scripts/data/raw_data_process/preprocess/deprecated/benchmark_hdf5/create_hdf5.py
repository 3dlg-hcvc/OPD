#!/usr/bin/env python3

# This file is only for validating the hdf5 function
# The code for converting the images into hdf5 is in the final_dataset.py

import argparse
import glob
import h5py
import numpy as np
import os

from PIL import Image

IN_DIR = "/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/MotionDataset_6.11/depth"
OUT_H5 = "/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/raw_data_process/preprocess/deprecated/benchmark_hdf5/test.hdf5"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create HDF5 file containing images")
    parser.add_argument("--in-dir", required=False, default=IN_DIR, help="input image directory")
    parser.add_argument("--out-h5", required=False, default=OUT_H5, help="output HDF5 file")
    parser.add_argument("--extension", default="png", help="image file extension")
    # The below argument is for testing two different storing method
    # This should not be used (This doesn't fully make use of hdf5 advantages)
    parser.add_argument(
        "--dataset-per-image", action="store_true", help="split each image into separate dataset"
    )
    args = parser.parse_args()

    # get all image files in directory
    files = sorted(glob.glob(args.in_dir + "/*." + args.extension))
    num_files = len(files)

    with h5py.File(args.out_h5, "a") as h5file:
        if args.dataset_per_image:
            # naively create one dataset per image
            for i, file in enumerate(files):
                path = os.path.relpath(file, start=args.in_dir)
                img = np.asarray(Image.open(file).convert("RGB"))
                dset = h5file.create_dataset(
                    path, data=img, compression="gzip", compression_opts=9, shuffle=True
                )
                if i % 1000 == 0:
                    print(f"{i}/{num_files}")
        else:
            # determine optimal chunk shape
            # first_img = img = np.asarray(Image.open(files[0]).convert("RGB"))
            first_img = img = np.asarray(Image.open(files[0]), dtype=np.float32)[:, :, None]
            img_shape = first_img.shape
            img_dtype = first_img.dtype
            dataset_shape = (num_files,) + img_shape
            chunk_shape = (1,) + img_shape
            string_dtype = h5py.string_dtype(encoding="utf-8")
            # create image dataset tensor
            dset_images = h5file.create_dataset(
                "images",
                shape=dataset_shape,
                dtype=img_dtype,
                chunks=chunk_shape,
                compression="gzip",
                compression_opts=9,
                shuffle=True,
            )
            # create image filenames dataset
            dset_filenames = h5file.create_dataset(
                "filenames", shape=(num_files,), dtype=string_dtype
            )
            # now fill the data
            for i in range(num_files):
                file = files[i]
                filepath = os.path.relpath(file, start=args.in_dir)
                # RGB
                # img = np.asarray(Image.open(file).convert("RGB"))
                # depth
                img = np.asarray(Image.open(file), dtype=np.float32)[:, :, None]
                dset_images[i] = img
                dset_filenames[i] = filepath
                if i % 1000 == 0:
                    print(f"{i}/{num_files}")
                if i == 1000:
                    break
