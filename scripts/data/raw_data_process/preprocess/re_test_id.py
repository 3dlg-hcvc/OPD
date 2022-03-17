import json

# Split the model ids into train/valid/test
VALIDMODELPATH = '../../sapien_data_process/valid_process/validModelPart.json'
old_TESTIDSPATH = './testIds_old.json'
old_VALIDIDPATH = './validIds_old.json'

if __name__ == "__main__":
    valid_model_file = open(VALIDMODELPATH)
    data = json.load(valid_model_file)
    valid_model_file.close()

    test_ids_file = open(old_TESTIDSPATH)
    test_ids_old = json.load(test_ids_file)
    test_ids_file.close()

    valid_ids_file = open(old_VALIDIDPATH)
    valid_ids_old = json.load(valid_ids_file)
    valid_ids_file.close()

    model_number_test_val = {}
    for model_cat in data.keys():
        model_number = 0
        for model_id in data[model_cat].keys():
            if model_id in test_ids_old or model_id in valid_ids_old:
                model_number += 1
        model_number_test_val[model_cat] = model_number

    test_ids = []
    valid_ids = []

    for model_cat in data.keys():
        model_num = model_number_test_val[model_cat]
        test_model_number = int(model_num * 0.5)

        current_test_number = 0
        current_valid_number = 0

        for model_id in data[model_cat].keys():
            if model_id not in test_ids_old and model_id not in valid_ids_old:
                # Don't change the model in the train set
                continue
            if(current_test_number < test_model_number):
                test_ids.append(model_id)
                current_test_number += 1
            else:
                valid_ids.append(model_id)
                current_valid_number += 1

    test_ids_file = open('./testIds.json', 'w')
    json.dump(test_ids, test_ids_file)
    test_ids_file.close()

    valid_ids_file = open('./validIds.json', 'w')
    json.dump(valid_ids, valid_ids_file)
    valid_ids_file.close()