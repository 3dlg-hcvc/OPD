import json

ValidPartPath = './validModelPart.json'

def get_part_number(data):
    for modelCat in data.keys():
        partNum = 0
        for id in data[modelCat].keys():
            partNum += len(data[modelCat][id].keys())
        print(modelCat, partNum)

    # part_num = {}
    # for modelCat in data.keys():
    #     for modelId in data[modelCat].keys():
    #         partNum = len(data[modelCat][modelId])
    #         if partNum not in part_num:
    #             part_num[partNum] = {'ModelNum': 0, 'ModelCat': {}}
            
    #         part_num[partNum]['ModelNum'] += 1
    #         if modelCat not in part_num[partNum]['ModelCat'].keys():
    #             part_num[partNum]['ModelCat'][modelCat] = 0
    #         part_num[partNum]['ModelCat'][modelCat]+= 1
    # print(part_num)

def get_structure_number(data):
    for modelCat in data.keys():
        print(modelCat, '!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        structure = {}
        for modelId in data[modelCat].keys():
            drawer = 0
            door = 0
            lid = 0
            for partId in data[modelCat][modelId].keys():
                if data[modelCat][modelId][partId] == 'drawer':
                    drawer += 1
                elif data[modelCat][modelId][partId] == 'door':
                    door += 1
                else:
                    lid += 1
            if (drawer, door, lid) not in structure:
                structure[(drawer, door, lid)] = 1
            else:
                structure[(drawer, door, lid)] += 1
        index = 1
        for i in structure:
            print(index, i, structure[i])
            index += 1

def get_model_number(data):
    model_num = 0
    part_num = 0
    image_num = 0
    for modelCat in data.keys():
        for modelId in data[modelCat].keys():
            model_num += 1
            model_part_num = len(data[modelCat][modelId].keys())
            image_num += model_part_num + 4*model_part_num**2
            for partId in data[modelCat][modelId].keys():
                part_num += 1

    print('Model num:', model_num)
    print('Part num:', part_num)
    print('Image num:', image_num)

if __name__ == "__main__":
    validPartFile = open(ValidPartPath)
    data = json.load(validPartFile)
    validPartFile.close()

    get_model_number(data)
    # get_structure_number(data)