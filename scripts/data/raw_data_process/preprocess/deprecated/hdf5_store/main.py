import re
import os
import cv2
import numpy as np
import h5py
import io
from PIL import Image
import argparse
import sys
import tables

def print_m(msg):
    msg_color = '\033[0m'
    sys.stdout.write(f'{msg_color}{msg}{msg_color}\n')


def print_w(warning):
    warn_color = '\033[93m'
    msg_color = '\033[0m'
    sys.stdout.write(f'{warn_color}Warning: {warning} {msg_color}\n')


def print_e(error):
    err_color = '\033[91m'
    msg_color = '\033[0m'
    sys.stdout.write(f'{err_color}Error: {error} {msg_color}\n')

def sorted_alphanum(file_list):
    """sort the file list by arrange the numbers in filenames in increasing order

    :param file_list: a file list
    :return: sorted file list
    """
    if len(file_list) <= 1:
        return file_list
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(file_list, key=alphanum_key)

def get_file_list(path, ext=''):
    if not os.path.exists(path):
        raise OSError('Path {} not exist!'.format(path))

    file_list = []
    for filename in os.listdir(path):
        file_ext = os.path.splitext(filename)[1]
        if (ext in file_ext or not ext) and os.path.isfile(os.path.join(path, filename)):
            file_list.append(os.path.join(path, filename))
    file_list = sorted_alphanum(file_list)
    return file_list

def get_folder_list(path, names=[]):
    if not os.path.exists(path):
        raise OSError('Path {} not exist!'.format(path))

    folder_list = []
    for foldername in os.listdir(path):
        if (foldername in names or not len(names)) and os.path.isdir(os.path.join(path, foldername)):
            folder_list.append(os.path.join(path, foldername))
    folder_list = sorted_alphanum(folder_list)
    return folder_list

def folder_exist(folder_path):
    if not os.path.exists(folder_path) or os.path.isfile(folder_path):
        return False
    else:
        return True

def bytes_size(nbytes):
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    i = 0
    while nbytes >= 1024 and i < len(suffixes)-1:
        nbytes /= 1024.
        i += 1
    f = ('%.2f' % nbytes).rstrip('0').rstrip('.')
    return '%s %s' % (f, suffixes[i])

def store_hdf5(file_path, img_paths):
    """ Stores images to an HDF5 file
        Parameters:
        ---------------
        file_path: path to output HDF5
        img_paths: list of paths to images
    """
    total = len(img_paths)
    # Create a new HDF5 file
    with h5py.File(file_path, "w") as hf:
        # for i in range(2):
        #     path = img_paths[i]
        for i, path in enumerate(img_paths):
            if args.verbose:
                print_m(f'save {i+1} / {total} image into hdf5')
            with open(path, 'rb') as img_f:  # open images as python binary
                binary_data = img_f.read()
            # image file name
            filename = os.path.basename(path)
            binary_data_np = np.asarray(binary_data)
            # Create a dataset in the file
            dataset = hf.create_dataset(
                name=filename,
                data=binary_data_np,
            )
    hdf5_name = os.path.basename(os.path.normpath(file_path))
    print_m('hdf5 {} file size: {} bytes'.format(hdf5_name, bytes_size(os.path.getsize(file_path))))

class ReadHdf5:
    def __init__(self, file_path, verbose=False):
        """ Reads images from HDF5
            Parameters:
            ---------------
            file_path: path to HDF5
            img_name: filename of image
        """
        self.file_path = file_path
        self.hf = None
        self.hf_mem = None
        self.v = verbose

    def open_hdf5(self):
        """ Open HDF5 file
        """
        if self.v:
            hdf5_name = os.path.basename(os.path.normpath(self.file_path))
            print_m('hdf5 {} file size: {} bytes'.format(hdf5_name, bytes_size(os.path.getsize(self.file_path))))

        self.hf = h5py.File(self.file_path, "r")
        self.hf_mem = tables.open_file(self.file_path, "r", driver="H5FD_CORE")
        
        if self.v:
            total = len(list(self.hf.keys()))
            print_m(f'{total} files in {hdf5_name}')

    def close_hdf5(self):
        self.hf.close()
        self.hf_mem.close()

    def read_hdf5(self, img_file, use_cv=False):
        """ Reads images from HDF5

            Returns:
            ----------
            image
        """
        import io
        buf = np.array(self.hf_mem.get_node('/'+img_file).read())
        buf = io.BytesIO(buf)
        # PIL image
        if not use_cv:
            img = Image.open(buf)
        # CV image
        else:
            file_bytes = np.asarray(bytearray(buf.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
            # print(img)
        return img

def configure(args):
    if not folder_exist(args.input):
        print_e(f'Input folder {args.input} not exists')
        return False
    
    if not folder_exist(os.path.basename(args.output)):
        print_e(f'Cannot open output folder {args.output}')
        return False

    return True

def store_all(args):
    print_m('storing images into hdf5 ...')
    root_folder = args.input
    output_folder = args.output
    path_list = get_folder_list(root_folder, args.select)

    for path in path_list:
        # folder contains the images
        folder = os.path.basename(os.path.normpath(path))
        img_paths = get_file_list(path, args.extension)
        
        hdf5_path = os.path.join(output_folder,folder)+'.h5'
        store_hdf5(hdf5_path, img_paths)

    print_m('store images finished!')

def read_test(args):
    root_folder = args.input
    output_folder = args.output
    path_list = get_folder_list(root_folder, args.select)

    for path in path_list:
        # folder contains the images
        folder = os.path.basename(os.path.normpath(path))
        img_paths = get_file_list(path, args.extension)
        
        hdf5_path = os.path.join(output_folder,folder)+'.h5'

        reader = ReadHdf5(hdf5_path)
        reader.open_hdf5()
        for i, img_path in enumerate(img_paths):
            if args.verbose:
                print_m('read {} / {}'.format(i+1, len(img_paths)))
            img = reader.read_hdf5(os.path.basename(img_path))
        reader.close_hdf5()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HDF5 images storage!')
    parser.add_argument('-in', '--input', dest='input', type=str, action='store', required=False,
                        default='/localhome/yma50/Development/motionnet/Dataset/MotionDataset_6.1',
                        help='Input root directory')
    parser.add_argument('-s', '--select', dest='select', type=str, nargs='+', default=['train', 'test', 'valid'], required=False,
                        help='select sub folders')
    parser.add_argument('-v', '--verbose', dest='verbose', default=False, action='store_true', required=False,
                        help='show step messages')
    parser.add_argument('-ext', '--extension', dest='extension', type=str, default='.png', required=False,
                        help='extension of storing images')
    parser.add_argument('-o', '--output', dest='output', type=str, action='store', required=False,
                        default='/localhome/yma50/Development/motionnet/TMP/raw_data_process/preprocess/hdf5_store/output',
                        help='Output hdf5 staging folder')
    parser.add_argument('--read_test', dest='read_test', default=False, action='store_true', required=False,
                        help='Does not write hdf5 files, but test the read functionality')

    args = parser.parse_args()

    if not configure(args):
        exit(0)

    if args.read_test:
        read_test(args)
    else:
        store_all(args)
