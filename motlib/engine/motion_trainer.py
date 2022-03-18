from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.evaluation import (
    DatasetEvaluator,
    print_csv_format,
)
from detectron2.solver import get_default_optimizer_params
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.evaluation import DatasetEvaluators
from detectron2.utils import comm
from motlib.data import MotionDatasetMapper
from motlib.evaluation import (
    MotionEvaluator,
    motion_inference_on_dataset,
)
import torch
import os
import logging
from collections import OrderedDict


# MotionNet Version: based on DefaultTrainer
class MotionTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        motion_dataset_mapper = MotionDatasetMapper(cfg, True)
        train_loader = build_detection_train_loader(cfg, mapper=motion_dataset_mapper)
        return train_loader

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        motion_dataset_mapper = MotionDatasetMapper(cfg, False)
        test_loader = build_detection_test_loader(
            cfg, dataset_name, mapper=motion_dataset_mapper
        )
        return test_loader

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Build an optimizer from config.
        """
        params = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
            weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
        )

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            return torch.optim.SGD(
                params,
                cfg.SOLVER.BASE_LR,
                momentum=cfg.SOLVER.MOMENTUM,
                nesterov=cfg.SOLVER.NESTEROV,
            )

        elif optimizer_type == "ADAM":
            return maybe_add_gradient_clipping(cfg, torch.optim.Adam)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Returns:
            DatasetEvaluator or None

        It is not implemented by default.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        # MotionNet: Evaluation for motion data (Including traditional coco evaluation)
        if "PART_CAT" in cfg.MODEL:
            part_cat = cfg.MODEL.PART_CAT
        else:
            part_cat = False

        if "AxisThres" in cfg.MODEL:
            AxisThres = cfg.MODEL.AxisThres
        else:
            AxisThres = 10

        if "OriginThres" in cfg.MODEL:
            OriginThres = cfg.MODEL.OriginThres
        else:
            OriginThres = 0.25

        if "IMAGESTATEPATH" in cfg.MODEL:
            image_state_path = cfg.MODEL.IMAGESTATEPATH
        else:
            image_state_path = None

        return MotionEvaluator(
            dataset_name,
            cfg,
            True,
            output_folder,
            motionnet_type=cfg.MODEL.MOTIONNET.TYPE,
            MODELATTRPATH=cfg.MODEL.MODELATTRPATH,
            PART_CAT=part_cat,
            AxisThres=AxisThres,
            OriginThres=OriginThres,
            motionstate=cfg.MODEL.MOTIONSTATE,
            image_state_path=image_state_path,
        )

    # Modify this function to support evaluating on the exsited inference file
    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            if "INFERENCE_FILE" in cfg:
                inference_file = cfg.INFERENCE_FILE
            else:
                inference_file = None
            results_i = motion_inference_on_dataset(
                model, data_loader, evaluator, inference_file
            )
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info(
                    "Evaluation results for {} in csv format:".format(dataset_name)
                )
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results
