#!/usr/bin/env python3

import argparse
import json
import os
import random
import shutil


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train motion net")
    parser.add_argument("--in-json", required=True, help="input annotations json file")
    parser.add_argument("--in-imgdir", required=True, help="input directory with images")
    parser.add_argument("--out-json", required=True, help="output annotations subset json")
    parser.add_argument("--out-imgdir", required=True, help="output images subset directory")
    parser.add_argument("--fraction", type=float, required=True, help="fraction to subset")
    args = parser.parse_args()

    # get all image ids, shuffle, and take subset
    ann_json = json.load(open(args.in_json))
    all_img_ids = [img["id"] for img in ann_json["images"]]
    random.shuffle(all_img_ids)
    all_num_ids = len(all_img_ids)
    num_ids = int(all_num_ids * args.fraction)
    img_ids = set(all_img_ids[:num_ids])

    # now filter annotations and images arrays (mutates ann_json inplace)
    annotations = [a for a in ann_json["annotations"] if a["image_id"] in img_ids]
    images = [i for i in ann_json["images"] if i["id"] in img_ids]
    ann_json["annotations"] = annotations
    ann_json["images"] = images

    # write new annotations json and copy images
    print(f'Writing {num_ids} img subset to {args.out_json} and {args.out_imgdir}')
    json.dump(ann_json, open(args.out_json, 'w'), separators=(',', ':'))
    os.makedirs(args.out_imgdir, exist_ok=True)

    for img in images:
        def copy_img(img_obj, img_key):
            basename = img_obj[img_key]
            src_img = os.path.join(args.in_imgdir, basename)
            dst_img = os.path.join(args.out_imgdir, basename)
            if not os.path.exists(src_img):
                print(f'[WARN] {src_img} does not exist; skipping...')
                return
            shutil.copy2(src_img, dst_img)
        copy_img(img, "file_name")
        copy_img(img, "depth_file_name")
