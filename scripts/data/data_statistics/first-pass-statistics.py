import json

PartPath = '/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/sapien_data_process/raw_process/careModelPart.json'

def get_model_number(data):
    model_num = 0
    part_num = 0
    for modelCat in data.keys():
        for modelId in data[modelCat].keys():
            model_num += 1
            part_num += len(data[modelCat][modelId])

    print('Model num:', model_num)
    print('Part num:', part_num)

if __name__ == "__main__":
    PartFile = open(PartPath)
    data = json.load(PartFile)
    PartFile.close()

    get_model_number(data)
    # get_structure_number(data)