import os
from time import time
import multiprocessing

RAWDATAPATH = "/local-scratch/localhome/hja40/Desktop/Dataset/raw_data_articulated_pose"
OUTPUTPATH = "/local-scratch/localhome/hja40/Desktop/Dataset/raw_data_6.2"


def existDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def copyModel(model_path):
    model_id = model_path.split("/")[-1]
    os.system(f'cp -r {model_path} {OUTPUTPATH}/')

if __name__ == "__main__":
    start = time()
    
    existDir(f'{OUTPUTPATH}')

    model_list = ['40453', '44962', '45132',
                    '45290', '46130', '46334',  '46462',
                    '46537', '46544', '46641', '47178', '47183',
                    '47296', '47233', '48010', '48253',  '48517',
                    '48740', '48876', '46230', '44853', '45135',
                    '45427', '45756', '46653', '46879', '47438', '47711', '48491', '46123',  '45841', '46440']
    
    pool = multiprocessing.Pool(processes=16)

    for model_id in model_list:
        print(f"Processing Model {model_id}")
        model_path = f"{RAWDATAPATH}/{model_id}"
        # convert(model_path)
        pool.apply_async(copyModel, (model_path,))

    pool.close()
    pool.join()

    stop = time()
    print(str(stop - start) + " seconds")
