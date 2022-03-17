import csv

annotation_path = "/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/real_scan_process/annotation.csv"

if __name__ == '__main__':
    file = open(annotation_path, "r")
    csv_reader = csv.reader(file)
    index = 0
    models = {}
    for line in csv_reader:
        if index == 0:
            index += 1
            continue
        model_index = line[4]
        model_cat = line[7]
        model_quality = line[8]

        if model_index not in models.keys():
            models[model_index] = [(model_cat, model_quality)]
        else:
            models[model_index].append((model_cat, model_quality))
        index += 1

    # Check if the same object has the same obj cat for different scans
    for model_ind in models.keys():
        cat = models[model_ind][0][0]
        for i in range(1, len(models[model_ind])):
            if not models[model_ind][i][0] == cat:
                print(f"ERROR: {model_ind}")
    
    # Get the number of valid objects
    num_valid_obj = 0
    num_valid_scan = 0
    valid_objs = []
    for model_ind in models.keys():
        valid = False
        for i in range(len(models[model_ind])):
            if not (models[model_ind][i][1]).lower() == "bad":
                valid = True
                num_valid_scan += 1
        if valid == True:
            num_valid_obj += 1
            valid_objs.append(model_ind)
    print(f"After annotating, there are {num_valid_obj} valid objects with {num_valid_scan} scans")

    # Statistics on the valid objs 
    model_cats = {}
    for model_ind in models.keys():
        if model_ind not in valid_objs:
            continue
        model_cat = models[model_ind][0][0]
        if model_cat not in model_cats.keys():
            model_cats[model_cat] = 1
        else:
            model_cats[model_cat] += 1

    print('\n')
    print("Model category statistics")
    print(model_cats)

    # Statistics on the valid scans
    scan_cats = {}
    for model_ind in models.keys():
        if model_ind not in valid_objs:
            continue
        for i in range(len(models[model_ind])):
            if not (models[model_ind][i][1]).lower() == "bad":
                scan_cat = models[model_ind][i][0]
                if scan_cat not in scan_cats:
                    scan_cats[scan_cat] = {"Good": 0, "Okay": 0, "Bad": 0}
                scan_cats[scan_cat][models[model_ind][i][1]] += 1

    print('\n')
    print("Scan category statistics")
    print(scan_cats)
