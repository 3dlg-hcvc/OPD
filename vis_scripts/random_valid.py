# This code is used to generate the image in the valid set for visualization (It will randomly pick several iamges)
import glob
import random
import json

VALIDATAPATH = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/MotionDataset_6.11/valid"
OUTPUTFILE = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/vis_scripts/valid_all.json"
random_seed = 518
image_num = 1000

if __name__ == "__main__":
    random.seed(random_seed)

    file_paths = glob.glob(f'{VALIDATAPATH}/*')
    files = [(file_path.split('/')[-1]).split('.')[0] for file_path in file_paths]
    random.shuffle(files)

    # if image_num equal to -1, then pick all images
    if image_num == -1:
        image_num = len(files)

    output_file = open(OUTPUTFILE, 'w')
    json.dump(files[:image_num], output_file)
    output_file.close()
