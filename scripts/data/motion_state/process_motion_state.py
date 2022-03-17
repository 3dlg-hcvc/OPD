import csv
import glob
import json
from time import time

raw_close_path = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/motion_state/raw_close.csv"
raw_data_path = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/raw_data_6.1"

model_close_path = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/motion_state/model_close.json"
image_close_path = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/motion_state/image_close.json"

if __name__ == '__main__':
    start = time()

    file = open(raw_close_path, "r")
    csv_reader = csv.reader(file)
    index = 0
    special_case = {}
    # Process the csv to get which part is special
    for line in csv_reader:
        if index == 0:
            index += 1
            continue
        model_name = line[0]
        raw_anno = line[1]
        if raw_anno == 'invalid':
            continue
        raw_part_annos = raw_anno.split(';')
        part_annos = {}
        for raw_part_anno in raw_part_annos:
            part_index = raw_part_anno.split('-')[0]
            part_annos[part_index] = raw_part_anno
        special_case[model_name] = part_annos

    model_close_value = {}
    dir_paths = glob.glob(f"{raw_data_path}/*")
    for current_dir in dir_paths:
        model_name = current_dir.split('/')[-1]
        # Load the default close state json
        with open(f"{current_dir}/origin_annotation/{model_name}-0-0.json") as f:
            motions = json.load(f)['motions']
        # Set the close state value to default rangeMin value
        part_close = {}
        for motion in motions:
            part_id = motion["partId"]
            part_close[part_id] = motion["rangeMin"]
        if model_name in special_case.keys():
            special_parts = special_case[model_name]
            for part_id in special_parts.keys():
                # Get the excat value for this part
                with open(f"{current_dir}/origin_annotation/{model_name}-{special_parts[part_id]}.json") as f:
                    motions = json.load(f)["motions"]
                for motion in motions:
                    if motion["partId"] == part_id:
                        part_close[part_id] = motion["value"]
                        break
        model_close_value[model_name] = part_close

    with open(model_close_path, "w") as f:
        json.dump(model_close_value, f)

    # Process to get the binary label and continuous value between the close state and current value for each image
    image_close_value = {}
    for current_dir in dir_paths:
        model_name = current_dir.split('/')[-1]
        part_close = model_close_value[model_name]
        rgb_paths = glob.glob(f"{current_dir}/origin/*.png")
        for rgb_path in rgb_paths:
            rgb_name = rgb_path.split('/')[-1].split('.')[0]
            with open(f"{current_dir}/origin_annotation/{rgb_name}.json") as f:
                motions = json.load(f)["motions"]
            image_close = {}
            for motion in motions:
                part_id = motion["partId"]
                image_close[part_id] = {}
                value = motion["value"]
                image_close[part_id]["close"] = (value == part_close[part_id])
                image_close[part_id]["value"] = abs(value - part_close[part_id])
            image_close_value[rgb_name] = image_close
    
    with open(image_close_path, "w") as f:
        json.dump(image_close_value, f)

    file.close()
    print(f"Total: {time() - start} seconds")