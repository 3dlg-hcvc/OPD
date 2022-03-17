import json
import glob

VALIDMODELPATH = '/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/sapien_data_process/valid_process/validModelPart.json'
RAWDATAPATH = "/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/raw_data_6.1/"

if __name__ == "__main__":
    valid_model_file = open(VALIDMODELPATH)
    data = json.load(valid_model_file)
    valid_model_file.close()

    image_count = {}
    for model_cat in data:
        image_count[model_cat] = 0
        for model_id in data[model_cat]:
            image_num = len(glob.glob(f"{RAWDATAPATH}/{model_id}/origin/*"))
            image_count[model_cat] += image_num
    
    print(image_count)

