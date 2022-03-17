# Script used to get the information form partnet_mobility dataset -> output  sapien_process.json
import glob
import json

DATA_DIR = '/home/hja40/Desktop/Dataset/data/models3d/partnetsim/mobility_v1_alpha5/'

if __name__ == "__main__":
    modelDirs = glob.glob(DATA_DIR + '*')

    sapienPro = {}

    for modelDir in modelDirs:
        modelId = modelDir.split('/')[-1]

        metaFile = open(f'{modelDir}/meta.json')
        modelCat = json.load(metaFile)['model_cat']
        metaFile.close()

        mobilityFile = open(f'{modelDir}/mobility_v2.json')
        modelMob = json.load(mobilityFile)
        parts = []
        for partMob in modelMob:
            partInfo = {}
            partInfo['id'] = partMob['id']
            partInfo['joint'] = partMob['joint']
            partInfo['name'] = partMob['name']
            parts.append(partInfo)

        modelInfo = {'modelId': modelId, 'parts': parts}
        if(modelCat not in sapienPro):
            sapienPro[modelCat] = [modelInfo]
        else:
            sapienPro[modelCat].append(modelInfo)
    
    sapienProFile = open('./sapien_process.json', 'w')
    json.dump(sapienPro, sapienProFile)
    sapienProFile.close()