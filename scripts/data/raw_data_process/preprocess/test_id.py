import json

# Split the model ids into train/valid/test
VALIDMODELPATH = '../../sapien_data_process/valid_process/validModelPart.json'
TESTRATIO = 0.25
VALIDRATIO = 0.05


if __name__ == "__main__":
    valid_model_file = open(VALIDMODELPATH)
    data = json.load(valid_model_file)
    valid_model_file.close()

    test_ids = []
    valid_ids = []

    for model_cat in data.keys():
        model_num = len(data[model_cat].keys())
        test_model_number = int(model_num * TESTRATIO)
        valid_model_number = int(model_num * VALIDRATIO)

        current_test_number = 0
        current_valid_number = 0

        for model_id in data[model_cat].keys():
            if(current_test_number < test_model_number):
                test_ids.append(model_id)
                current_test_number += 1
            elif(current_valid_number < valid_model_number):
                valid_ids.append(model_id)
                current_valid_number += 1
            else:
                break

    test_ids_file = open('./testIds.json', 'w')
    json.dump(test_ids, test_ids_file)
    test_ids_file.close()

    valid_ids_file = open('./validIds.json', 'w')
    json.dump(valid_ids, valid_ids_file)
    valid_ids_file.close()