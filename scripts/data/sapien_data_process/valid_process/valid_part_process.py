import json

validPartPath = './validPart.txt'

if __name__ == "__main__":
    validPartFile = open(validPartPath)
    data = validPartFile.read()
    validPartFile.close()
    
    validPart = {}

    # Process the txt data and convert it into json with the similar format as careModelPart.json
    lines = data.split('\n')
    for line in lines:
        processLine = line.split(' ')
        modelCat = processLine[0]
        if modelCat == 'Door':
            continue
        modelId = processLine[1]
        partIndex = processLine[2]
        partLabel = processLine[3]

        if(modelCat not in validPart.keys()):
            validPart[modelCat] = {}
        if(modelId not in validPart[modelCat].keys()):
            validPart[modelCat][modelId] = {}
        validPart[modelCat][modelId][partIndex] = partLabel

    validPartFile = open('./validModelPart.json', 'w')
    json.dump(validPart, validPartFile)
    validPartFile.close()
