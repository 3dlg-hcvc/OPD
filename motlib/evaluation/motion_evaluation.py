# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import pickle
from collections import OrderedDict, defaultdict
from contextlib import ExitStack
import time
import datetime

from torch._C import Value
import pycocotools.mask as mask_util
import torch
from torch import nn
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.utils.comm import get_world_size
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.logger import create_small_table, log_every_n_seconds
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_context,
)

from .motion_coco_eval import MotionCocoEval


# MotionNet: modify based on DensePoseCOCOEvaluation
class MotionEvaluator(DatasetEvaluator):
    # Load ground truth from the annotation json file
    def __init__(
        self,
        dataset_name,
        cfg,
        distributed,
        output_dir=None,
        motionnet_type="BMCC",
        MODELATTRPATH=None,
        TYPE_MATCH=False,
        PART_CAT=False,
        MICRO_AVG=False,
        AxisThres=10,
        OriginThres=0.25,
        motionstate=False,
        image_state_path=None,
    ):
        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        json_file = PathManager.get_local_path(self._metadata.json_file)

        # Filter the image based on the filter file
        if "FILTER_FILE" in cfg and not cfg.FILTER_FILE == None:
            with open(cfg.FILTER_FILE, "r") as f:
                selections = json.load(f)
            dataset = json.load(open(json_file, "r"))
            new_dataset = {
                "annotations": [],
                "categories": dataset["categories"],
                "images": [],
                "info": dataset["info"],
                "licenses": dataset["licenses"],
            }
            useful_image_ids = []
            for img in dataset["images"]:
                if img["file_name"] in selections:
                    new_dataset["images"].append(img)
                    useful_image_ids.append(img["id"])
            for anno in dataset["annotations"]:
                if anno["image_id"] in useful_image_ids:
                    new_dataset["annotations"].append(anno)
            self._coco_api = COCO()
            self._coco_api.dataset = new_dataset
            self._coco_api.createIndex()
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                self._coco_api = COCO(json_file)

        self.motionnet_type = motionnet_type
        self.MODELATTRPATH = MODELATTRPATH
        self.TYPE_MATCH = TYPE_MATCH
        self.PART_CAT = PART_CAT
        self.MICRO_AVG = MICRO_AVG
        self.AxisThres = AxisThres
        self.OriginThres = OriginThres
        self.motionstate = motionstate
        self.image_state_path = image_state_path

    def reset(self):
        self._predictions = []

    def _tasks_from_config(self, cfg):
        """
        Returns:
            tuple[str]: tasks that can be evaluated under the given configuration.
        """
        tasks = ("bbox",)
        if cfg.MODEL.MASK_ON:
            tasks = tasks + ("segm",)
        if cfg.MODEL.KEYPOINT_ON:
            tasks = tasks + ("keypoints",)
        return tasks

    def process(self, inputs, outputs):
        # Process the raw predictions into json format
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            instances = output["instances"].to(self._cpu_device)
            prediction["instances"] = prediction_to_json(instances, input["image_id"], motionstate=self.motionstate)
            self._predictions.append(prediction)

    def load_from_file(self, inference_file_path):
        with PathManager.open(inference_file_path, "rb") as f:
            buffer = io.BytesIO(f.read())
        self._predictions = torch.load(buffer)

    def evaluate(self):
        # Deal with the distributed
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        # Save all the predictions
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        self._eval_predictions(set(self._tasks), predictions)

        # Based on densepose
        return copy.deepcopy(self._results)

    def _eval_predictions(self, tasks, predictions):
        """
        Evaluate predictions on densepose.
        Return results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k
                for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in coco_results:
                category_id = result["category_id"]
                assert (
                    category_id in reverse_id_mapping
                ), "A prediction has category_id={}, which is not available in the dataset.".format(
                    category_id
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_motion_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        self._logger.info("Evaluating predictions ...")
        for task in sorted(tasks):
            coco_eval = (
                _evaluate_predictions_on_motion(
                    self._coco_api,
                    coco_results,
                    task,
                    motionnet_type=self.motionnet_type,
                    MODELATTRPATH=self.MODELATTRPATH,
                    TYPE_MATCH=self.TYPE_MATCH,
                    PART_CAT=self.PART_CAT,
                    MICRO_AVG=self.MICRO_AVG,
                    AxisThres=self.AxisThres,
                    OriginThres=self.OriginThres,
                    motionstate=self.motionstate,
                    image_state_path=self.image_state_path,
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            self._results[task] = res

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """
        # MotionNet: pc: the mAP is based on part category; mt: the mAP is based on motion type
        metrics = {
            "bbox": [
                "AP_pc",
                "AP50_pc",
                "AP75_pc",
                "all_motion50_pc",
                "motion_type50_mt",
                "motion_origin50_mt",
                "motion_axis50_mt",
            ],
            "segm": [
                "AP_pc",
                "AP50_pc",
                "AP75_pc",
                "all_motion50_pc",
                "motion_type50_mt",
                "motion_origin50_mt",
                "motion_axis50_mt",
            ],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(
                coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan"
            )
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type)
            + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]

        # precision has dims (iou, recall, cls, area range, max dets)
        # assert len(class_names) == precisions.shape[2]

        # results_per_category = []
        # for idx, name in enumerate(class_names):
        #     # area range index 0: all area ranges
        #     # max dets index -1: typically 100 per image
        #     precision = precisions[:, :, idx, 0, -1]
        #     precision = precision[precision > -1]
        #     ap = np.mean(precision) if precision.size else float("nan")
        #     results_per_category.append(("{}".format(name), float(ap * 100)))

        # # tabulate it
        # N_COLS = min(6, len(results_per_category) * 2)
        # results_flatten = list(itertools.chain(*results_per_category))
        # results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        # table = tabulate(
        #     results_2d,
        #     tablefmt="pipe",
        #     floatfmt=".3f",
        #     headers=["category", "AP"] * (N_COLS // 2),
        #     numalign="left",
        # )
        # self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        # results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results


# MotionNet: based on instances_to_coco_json and relevant codes in densepose
def prediction_to_json(instances, img_id, motionstate=False):
    """
    Args:
        instances (Instances): the output of the model
        img_id (str): the image id in COCO

    Returns:
        list[dict]: the results in densepose evaluation format
    """
    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()
    # Prediction for MotionNet
    # mtype = instances.mtype.squeeze(axis=1).tolist()

    # 2.0.3
    if instances.has("pdim"):
        pdim = instances.pdim.tolist()
    if instances.has("ptrans"):
        ptrans = instances.ptrans.tolist()
    if instances.has("prot"):
        prot = instances.prot.tolist()

    mtype = instances.mtype.tolist()
    morigin = instances.morigin.tolist()
    maxis = instances.maxis.tolist()
    # TODO(AXC): check if mextrinsic exists
    if instances.has("mextrinsic"):
        mextrinsic = instances.mextrinsic.tolist()

    if motionstate:
        mstate = instances.mstate.tolist()

    # MotionNet has masks in the annotation
    # use RLE to encode the masks, because they are too large and takes memory
    # since this evaluator stores outputs of the entire dataset
    rles = [
        mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
        for mask in instances.pred_masks
    ]
    for rle in rles:
        # "counts" is an array encoded by mask_util as a byte-stream. Python3's
        # json writer which always produces strings cannot serialize a bytestream
        # unless you decode it. Thankfully, utf-8 works out (which is also what
        # the pycocotools/_mask.pyx does).
        rle["counts"] = rle["counts"].decode("utf-8")

    results = []
    for k in range(len(instances)):
        if instances.has("pdim"):
            result = {
                "image_id": img_id,
                "category_id": classes[k],
                "bbox": boxes[k],
                "score": scores[k],
                "segmentation": rles[k],
                "pdim": pdim[k],
                "ptrans": ptrans[k],
                "prot": prot[k],
                "mtype": mtype[k],
                "morigin": morigin[k],
                "maxis": maxis[k],
            }
        elif instances.has("mextrinsic"):
            result = {
                "image_id": img_id,
                "category_id": classes[k],
                "bbox": boxes[k],
                "score": scores[k],
                "segmentation": rles[k],
                "mtype": mtype[k],
                "morigin": morigin[k],
                "maxis": maxis[k],
                # TODO(AXC): check if mextrinsic exists
                "mextrinsic": mextrinsic[k],
            }
        else:
            result = {
                "image_id": img_id,
                "category_id": classes[k],
                "bbox": boxes[k],
                "score": scores[k],
                "segmentation": rles[k],
                "mtype": mtype[k],
                "morigin": morigin[k],
                "maxis": maxis[k],
            }
        if motionstate:
            result["mstate"] = mstate[k]
        results.append(result)
    return results


# MotionNet: based on _evaluate_predictions_on_coco for CoCo and densepose
def _evaluate_predictions_on_motion(
    coco_gt,
    coco_results,
    iou_type,
    kpt_oks_sigmas=None,
    motionnet_type="BMCC",
    MODELATTRPATH=None,
    TYPE_MATCH=False,
    PART_CAT=False,
    MICRO_AVG=False,
    AxisThres=10,
    OriginThres=0.25,
    motionstate=False,
    image_state_path=None,
):
    """
    Evaluate the coco results using COCOEval API.
    """
    # Copy the below two things because our mAP based on motion type will change them
    coco_gt = copy.deepcopy(coco_gt)
    coco_results = copy.deepcopy(coco_results)

    if iou_type == "segm":
        # coco_results = copy.deepcopy(coco_results)
        # When evaluating mask AP, if the results contain bbox, cocoapi will
        # use the box area as the area of the instance, instead of the mask area.
        # This leads to a different definition of small/medium/large.
        # We remove the bbox field to let mask AP use mask area.
        for c in coco_results:
            c.pop("bbox", None)
    coco_dt = coco_gt.loadRes(coco_results)
    print("########### Results on the part category ###############")
    coco_eval = MotionCocoEval(
        coco_gt,
        coco_dt,
        iou_type,
        motionnet_type=motionnet_type,
        MODELATTRPATH=MODELATTRPATH,
        TYPE_MATCH=TYPE_MATCH,
        PART_CAT=PART_CAT,
        MICRO_AVG=MICRO_AVG,
        MACRO_TYPE=False,
        AxisThres=AxisThres,
        OriginThres=OriginThres,
        motionstate=motionstate,
        image_state_path=image_state_path,
    )

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # To get the results for macro average over motion type
    MOTION_TYPE = {"rotation": 0, "translation": 1}
    # Modify the coco_gt firstly
    coco_gt.cats = {
        1: {"id": 1, "name": "rotation", "supercategory": "Container_parts"},
        2: {"id": 2, "name": "translation", "supercategory": "Container_parts"},
    }
    coco_gt.catToImgs = defaultdict(list)
    for id in coco_gt.anns.keys():
        # The category id starts from 1
        motion_type = MOTION_TYPE[coco_gt.anns[id]["motion"]["type"]] + 1
        coco_gt.anns[id]["category_id"] = motion_type
        coco_gt.catToImgs[motion_type].append(coco_gt.anns[id]["image_id"])
    # Modify the coco_dt
    coco_dt.cats = {
        1: {"id": 1, "name": "rotation", "supercategory": "Container_parts"},
        2: {"id": 2, "name": "translation", "supercategory": "Container_parts"},
    }
    coco_dt.catToImgs = defaultdict(list)
    for id in coco_dt.anns.keys():
        motion_type = int(coco_dt.anns[id]["mtype"]) + 1
        coco_dt.anns[id]["category_id"] = motion_type
        coco_dt.catToImgs[motion_type].append(coco_dt.anns[id]["image_id"])
    print("########### Results on the motion type ###############")
    macro_eval_mtype = MotionCocoEval(
        coco_gt,
        coco_dt,
        iou_type,
        motionnet_type=motionnet_type,
        MODELATTRPATH=MODELATTRPATH,
        MACRO_TYPE=True,
        AxisThres=AxisThres,
        OriginThres=OriginThres,
        motionstate=motionstate,
        image_state_path=image_state_path,
    )

    macro_eval_mtype.evaluate()
    macro_eval_mtype.accumulate()
    macro_eval_mtype.summarize()

    # Update the stats for the motion type/origin/axis based on motion type
    coco_eval.stats[4] = macro_eval_mtype.stats[4]
    coco_eval.stats[5] = macro_eval_mtype.stats[5]
    coco_eval.stats[6] = macro_eval_mtype.stats[6]

    return coco_eval
    # return macro_eval_mtype


# Modify this function to support evaluating on existed inference file
def motion_inference_on_dataset(
    model, data_loader, evaluator, inference_file_path=None, filter_file_path=None
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use `None` if you only want
            to benchmark, but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        assert ValueError("Motion Error: the evaluator is not corretly defined")
    evaluator.reset()

    selections = None
    if not filter_file_path == None:
        with open(filter_file_path, "r") as f:
            selections = json.load(f)

    if inference_file_path == None:
        num_warmup = min(5, total - 1)
        start_time = time.perf_counter()
        total_compute_time = 0
        with ExitStack() as stack:
            if isinstance(model, nn.Module):
                stack.enter_context(inference_context(model))
            stack.enter_context(torch.no_grad())

            for idx, inputs in enumerate(data_loader):
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_compute_time = 0

                if not selections == None:
                    # When inferencing, the batch size is always 1
                    image_name = inputs[0]["file_name"].split("/")[-1]
                    if image_name not in selections:
                        continue
                start_compute_time = time.perf_counter()
                outputs = model(inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_compute_time += time.perf_counter() - start_compute_time
                evaluator.process(inputs, outputs)

                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                seconds_per_img = total_compute_time / iters_after_start
                if idx >= num_warmup * 2 or seconds_per_img > 5:
                    total_seconds_per_img = (
                        time.perf_counter() - start_time
                    ) / iters_after_start
                    eta = datetime.timedelta(
                        seconds=int(total_seconds_per_img * (total - idx - 1))
                    )
                    log_every_n_seconds(
                        logging.INFO,
                        "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                            idx + 1, total, seconds_per_img, str(eta)
                        ),
                        n=5,
                    )

        # Measure the time only for this worker (before the synchronization barrier)
        total_time = time.perf_counter() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        # NOTE this format is parsed by grep
        logger.info(
            "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
                total_time_str, total_time / (total - num_warmup), num_devices
            )
        )
        total_compute_time_str = str(
            datetime.timedelta(seconds=int(total_compute_time))
        )
        logger.info(
            "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
                total_compute_time_str,
                total_compute_time / (total - num_warmup),
                num_devices,
            )
        )
    else:
        evaluator.load_from_file(inference_file_path)

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results
