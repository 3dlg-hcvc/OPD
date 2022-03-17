import contextlib
import io
import logging
import os
import pycocotools.mask as mask_util

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

logger = logging.getLogger(__name__)


def register_motion_instances(name, metadata, json_file, image_root):
    """
    Register MotionNet dataset.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root

    DatasetCatalog.register(name, lambda: load_motion_json(json_file, image_root, name))

    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, **metadata
    )


# Adapted detectron2.data.datasets.coco.load_coco_json to add custom img annotations
def load_motion_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file with COCO instances annotation format + motionnet hacks

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO

    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    imgs_anns = list(zip(imgs, anns))

    logger.info(f"Loaded {len(imgs_anns)} images in COCO format from {json_file}")

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id", "motion"] + (extra_annotation_keys or [])

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]
        # 2DMotion changes: add extra annotations from img_dict
        # Ideally, this would be handled with something like "extra_img_keys"
        record["camera"] = img_dict["camera"]
        record["depth_file_name"] = img_dict["depth_file_name"]
        record["label"] = img_dict["label"]

        objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:  # filter out invalid polygons (< 3 points)
                    segm = [p for p in segm if len(p) % 2 == 0 and len(p) >= 6]
                    if len(segm) == 0:
                        continue  # ignore this instance
                obj["segmentation"] = segm
            
            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                obj["category_id"] = id_map[obj["category_id"]]
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts
