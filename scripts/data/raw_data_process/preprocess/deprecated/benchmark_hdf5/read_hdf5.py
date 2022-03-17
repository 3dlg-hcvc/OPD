#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
from timeit import default_timer as timer
from PIL import Image

IN_H5 = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/MotionDataset_h5_6.11/depth.h5"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark reading HDF5 with image data")
    parser.add_argument("--in-h5", required=False, default=IN_H5, help="input HDF5 file")
    # The below argument is for testing two different storing method
    # This should not be used (This doesn't fully make use of hdf5 advantages)
    parser.add_argument(
        "--dataset-per-image",
        action="store_true",
        help="whether HDF5 has separate dataset per image",
    )
    args = parser.parse_args()

    start_time = timer()
    all_img_max = 0
    with h5py.File(args.in_h5, "r") as h5file:
        if args.dataset_per_image:
            for name in h5file:
                img = h5file[name]
                all_img_max = max(np.amax(img), all_img_max)
        else:
            dset_images = h5file["depth_filenames"]
            # num_images = len(dset_images)
            num_images = dset_images.shape
            print(dset_images[0])
            # for i in range(num_images):
            #     img = dset_images[i]
            #     all_img_max = max(np.amax(img), all_img_max)
            # print(dset_images[0][:, :, 0].max())
    end_time = timer()

    print(f"Max image pixel value: {all_img_max}")
    print(f"Total time taken to read HDF5 file: {end_time - start_time}")
