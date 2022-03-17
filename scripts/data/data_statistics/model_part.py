import json

VALIDMODELPATH = '/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/sapien_data_process/valid_process/validModelPart.json'
TESTIDSPATH = '/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/raw_data_process/preprocess/testIds.json'
VALIDIDPATH = '/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/raw_data_process/preprocess/validIds.json'

if __name__ == "__main__":
    valid_model_file = open(VALIDMODELPATH)
    data = json.load(valid_model_file)
    valid_model_file.close()

    test_ids_file = open(TESTIDSPATH)
    test_ids = json.load(test_ids_file)
    test_ids_file.close()

    valid_ids_file = open(VALIDIDPATH)
    valid_ids = json.load(valid_ids_file)
    valid_ids_file.close()

    model_part = {'train': {"model": 0, "part": 0, "drawer": 0, "door": 0, "lid": 0}, 'val': {"model": 0, "part": 0, "drawer": 0, "door": 0, "lid": 0}, 'test': {"model": 0, "part": 0, "drawer": 0, "door": 0, "lid": 0}}
    for model_cat in data.keys():
        for model_id in data[model_cat].keys():
            if model_id in test_ids:
                set_name = "test"
            elif model_id in valid_ids:
                set_name = "val"
            else:
                set_name = "train"
            model_part[set_name]['model'] += 1
            model_part[set_name]['part'] += len(data[model_cat][model_id])
            for v in data[model_cat][model_id].values():
                model_part[set_name][v.strip()] += 1

    print(model_part)