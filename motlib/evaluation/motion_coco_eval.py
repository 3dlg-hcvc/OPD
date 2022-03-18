import numpy as np
from numpy import dot
from numpy.linalg import norm
import datetime
import time
from collections import defaultdict
from pycocotools import mask as maskUtils
import copy
import json

MOTION_TYPE = {"rotation": 0, "translation": 1}

# Calculate the degree between two axis, return (0, pi)
def getDegreeFromAxes(ax1, ax2):
    cosvalue = dot(ax1, ax2) / (norm(ax1) * norm(ax2))
    cosvalue = min(cosvalue, 1.0)
    cosvalue = max(cosvalue, -1.0)
    degree = np.arccos(cosvalue) / np.pi * 180
    return degree


# Calculates Rotation Matrix given euler angles (ZYX).
def ZYX2Mat(theta):

    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta[0]), -np.sin(theta[0])],
            [0, np.sin(theta[0]), np.cos(theta[0])],
        ]
    )
    R_y = np.array(
        [
            [np.cos(theta[1]), 0, np.sin(theta[1])],
            [0, 1, 0],
            [-np.sin(theta[1]), 0, np.cos(theta[1])],
        ]
    )
    R_z = np.array(
        [
            [np.cos(theta[2]), -np.sin(theta[2]), 0],
            [np.sin(theta[2]), np.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )
    R = np.dot(R_z, np.dot(R_y, R_x))

    return R

# MotionNet: based on cocoeval in pycocotools
class MotionCocoEval:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(self, cocoGt=None, cocoDt=None, iouType="segm", motionnet_type="BMCC", MODELATTRPATH=None, PART_CAT=False, MACRO_TYPE=False, AxisThres=10, OriginThres=0.25, motionstate=False, image_state_path=None):
        """
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        """
        if not iouType:
            print("iouType not specified. use default iouType segm")
        self.cocoGt = cocoGt  # ground truth COCO API
        self.cocoDt = cocoDt  # detections COCO API
        self.evalImgs = defaultdict(
            list
        )  # per-image per-category evaluation results [KxAxI] elements
        self.eval = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        self.params = Params(iouType=iouType)  # parameters

        self._paramsEval = {}  # parameters for evaluation
        self.stats = []  # result summarization
        self.ious = {}  # ious between all gts and dts

        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

        self.motionnet_type = motionnet_type
        self.MODELATTRPATH = MODELATTRPATH
        self.diagonal_length = self.getDiagLength()
        self.PART_CAT = PART_CAT
        self.MACRO_TYPE = MACRO_TYPE
        self.AxisThres = AxisThres
        self.OriginThres = OriginThres
        self.motionstate = motionstate
        self.image_state_path = image_state_path
        if self.motionstate:
            if self.image_state_path == None:
                raise ValueError("No image state file")
            with open(self.image_state_path, "r") as f:
                self.image_states = json.load(f)


    def getDiagLength(self):
        model_attr_file = open(self.MODELATTRPATH)
        model_bound = json.load(model_attr_file)
        model_attr_file.close()
        # The data on the urdf need a coordinate transform [x, y, z] -> [z, x, y]
        diagonal_length = {}
        for model in model_bound:
            diagonal_length[model] = model_bound[model]["diameter"]
        return diagonal_length

    def _prepare(self):
        """
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        """

        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann["segmentation"] = rle

        p = self.params
        if p.useCats:
            gts = self.cocoGt.loadAnns(
                self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)
            )
            dts = self.cocoDt.loadAnns(
                self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)
            )
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == "segm":
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            if self.motionnet_type == "BMOC":
                camera = self.cocoGt.loadImgs(gt["image_id"])[0]["camera"]
                extrinsic = camera["extrinsic"]["matrix"]
                gt["gt_extrinsic"] = extrinsic
            gt["ignore"] = gt["ignore"] if "ignore" in gt else 0
            gt["ignore"] = "iscrowd" in gt and gt["iscrowd"]
            if p.iouType == "keypoints":
                gt["ignore"] = (gt["num_keypoints"] == 0) or gt["ignore"]
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        for gt in gts:
            self._gts[gt["image_id"], gt["category_id"]].append(gt)
        for dt in dts:
            self._dts[dt["image_id"], dt["category_id"]].append(dt)

        self.evalImgs = defaultdict(list)  # per-image per-category evaluation results
        self.eval = {}  # accumulated evaluation results

    def evaluate(self):
        """
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        """
        tic = time.time()
        print("Running per image Motion evaluation...")
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = "segm" if p.useSegm == 1 else "bbox"
            print(
                "useSegm (deprecated) is not None. Running {} evaluation".format(
                    p.iouType
                )
            )
        print("Evaluate annotation type *{}*".format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == "segm" or p.iouType == "bbox":
            computeIoU = self.computeIoU
        elif p.iouType == "keypoints":
            computeIoU = self.computeOks
        self.ious = {
            (imgId, catId): computeIoU(imgId, catId)
            for imgId in p.imgIds
            for catId in catIds
        }

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [
            evaluateImg(imgId, catId, areaRng, maxDet)
            for catId in catIds
            for areaRng in p.areaRng
            for imgId in p.imgIds
        ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print("DONE (t={:0.2f}s).".format(toc - tic))

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0 : p.maxDets[-1]]

        if p.iouType == "segm":
            g = [g["segmentation"] for g in gt]
            d = [d["segmentation"] for d in dt]
        elif p.iouType == "bbox":
            g = [g["bbox"] for g in gt]
            d = [d["bbox"] for d in dt]
        else:
            raise Exception("unknown iouType for iou computation")

        # compute iou between each dt and gt region
        iscrowd = [int(o["iscrowd"]) for o in gt]
        ious = maskUtils.iou(d, g, iscrowd)
        return ious

    def computeOks(self, imgId, catId):
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d["score"] for d in dts], kind="mergesort")
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0 : p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = p.kpt_oks_sigmas
        vars = (sigmas * 2) ** 2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt["keypoints"])
            xg = g[0::3]
            yg = g[1::3]
            vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt["bbox"]
            x0 = bb[0] - bb[2]
            x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]
            y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt["keypoints"])
                xd = d[0::3]
                yd = d[1::3]
                if k1 > 0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0 - xd), axis=0) + np.max((z, xd - x1), axis=0)
                    dy = np.max((z, y0 - yd), axis=0) + np.max((z, yd - y1), axis=0)
                e = (dx ** 2 + dy ** 2) / vars / (gt["area"] + np.spacing(1)) / 2
                if k1 > 0:
                    e = e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        """
        perform evaluation for single category and image
        :return: dict (single image results)
        """
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if g["ignore"] or (g["area"] < aRng[0] or g["area"] > aRng[1]):
                g["_ignore"] = 1
            else:
                g["_ignore"] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o["iscrowd"]) for o in gt]
        # load computed ious
        ious = (
            self.ious[imgId, catId][:, gtind]
            if len(self.ious[imgId, catId]) > 0
            else self.ious[imgId, catId]
        )

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g["_ignore"] for g in gt])
        gtType = np.array([MOTION_TYPE[g["motion"]["type"]] for g in gt])
        dtIg = np.zeros((T, D))
        # MotionNet
        if self.motionnet_type == "BMOC":
            dtOriginWorld = np.zeros((T, D))
            dtAxisWorld = np.zeros((T, D))
        if self.motionstate:
            dtState = np.zeros((T, D))
        dtType = np.zeros((T, D))
        rotType = np.zeros((T, D))
        dtOrigin = np.zeros((T, D))
        dtOriginThres = np.zeros((T, D))
        dtAxis = np.zeros((T, D))
        dtAxisThres = np.zeros((T, D))
        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    rotType[tind, dind] = int(d["mtype"])
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]["id"]
                    gtm[tind, m] = d["id"]

                    model_name = self.cocoGt.loadImgs(int(gt[m]["image_id"]))[0]['file_name'].split('-')[0]
                    diagonal_length = self.diagonal_length[model_name]

                    if self.motionstate:
                        image_name = self.cocoGt.loadImgs(int(gt[m]["image_id"]))[0]['file_name'].split('.')[0]
                        part_id = gt[m]["motion"]["partId"]
                        gt_state = int(self.image_states[image_name][part_id]["close"])
                        pred_state = int(d["mstate"])
                        if gt_state == pred_state:
                            dtState[tind, dind] = 1
                        else:
                            dtState[tind, dind] = 0

                    # Calculate the type value
                    gt_type = MOTION_TYPE[gt[m]["motion"]["type"]]
                    pred_type = int(d["mtype"])
                    # dtType used for both MT and mTP
                    if gt_type == pred_type:
                        dtType[tind, dind] = 1
                    else:
                        dtType[tind, dind] = 0

                    # Calculate the distance between predicted origin and the gt axis line
                    # Calculate the distance between origin (smaller is better)
                    # gt_origin = gt[m]["motion"]["initial_origin"]
                    # origin_distance = (
                    #     (d["morigin"][0] - gt_origin[0]) ** 2
                    #     + (d["morigin"][1] - gt_origin[1]) ** 2
                    #     + (d["morigin"][2] - gt_origin[2]) ** 2
                    # ) ** 0.5
                    # # dtOrigin[tind, dind] = 2 - 2 * (1 / (1 + np.exp(-origin_distance)))
                    # dtOrigin[tind, dind] = origin_distance
                    # # Calculate the cosine similarity
                    # gt_axis = np.array(gt[m]["motion"]["initial_axis"])
                    # pre_axis = np.array(d["maxis"])
                    # dtAxis[tind, dind] = dot(gt_axis, pre_axis) / (
                    #     norm(gt_axis) * norm(pre_axis)

                    pred_origin = np.array(d["morigin"])
                    gt_origin = np.array(gt[m]["motion"]["current_origin"])
                    pred_axis = np.array(d["maxis"])
                    gt_axis = np.array(gt[m]["motion"]["current_axis"])

                    # Evaluation metric in the object coordinate for BMOC
                    if self.motionnet_type == "BMOC":
                        # gt_extrinsic: 16; pred_extrinsic: 12
                        gt_extrinsic = np.array(gt[m]["gt_extrinsic"])
                        gt_extrinsic_mat = np.reshape(gt_extrinsic, (4, 4)).T
                        pred_extrinsic = np.array(d["mextrinsic"])
                        pred_extrinsic_all = np.zeros(16)
                        pred_extrinsic_all[-1] = 1
                        pred_extrinsic_all[0:3] = pred_extrinsic[0:3]
                        pred_extrinsic_all[4:7] = pred_extrinsic[3:6]
                        pred_extrinsic_all[8:11] = pred_extrinsic[6:9]
                        pred_extrinsic_all[12:15] = pred_extrinsic[9:12]
                        pred_extrinsic_mat = np.reshape(pred_extrinsic_all, (4, 4)).T

                        gt_origin_world = np.dot(
                            gt_extrinsic_mat, np.array(list(gt_origin) + [1])
                        )[0:3]
                        gt_axis_end_cam = gt_origin + gt_axis
                        gt_axis_end_world = np.dot(
                            gt_extrinsic_mat, np.array(list(gt_axis_end_cam) + [1])
                        )[0:3]
                        gt_axis_world = gt_axis_end_world - gt_origin_world

                        pred_origin_world = np.dot(
                            pred_extrinsic_mat, np.array(list(pred_origin) + [1])
                        )[0:3]
                        pred_axis_end_cam = pred_origin + pred_axis
                        pred_axis_end_world = np.dot(
                            pred_extrinsic_mat, np.array(list(pred_axis_end_cam) + [1])
                        )[0:3]
                        pred_axis_world = pred_axis_end_world - pred_origin_world

                        p_world = pred_origin_world - gt_origin_world
                        dtOriginWorld[tind, dind] = np.linalg.norm(
                            np.cross(p_world, gt_axis_world)
                        ) / np.linalg.norm(gt_axis_world) / diagonal_length

                        dtAxisWorld[tind, dind] = dot(
                            gt_axis_world, pred_axis_world
                        ) / (norm(gt_axis_world) * norm(pred_axis_world))
                        if dtAxisWorld[tind, dind] < 0:
                            dtAxisWorld[tind, dind] = -dtAxisWorld[tind, dind]
                        dtAxisWorld[tind, dind] = min(dtAxisWorld[tind, dind], 1.0)
                        ## dtAxis used for evaluation metric MD
                        dtAxisWorld[tind, dind] = (
                            np.arccos(dtAxisWorld[tind, dind]) / np.pi * 180
                        )

                    p = pred_origin - gt_origin
                    ## dtOrigin used for evaluation metric MO (matched origin)
                    dtOrigin[tind, dind] = np.linalg.norm(
                        np.cross(p, gt_axis)
                    ) / np.linalg.norm(gt_axis) / diagonal_length
                    ## dtOriginThres used for evaluation metric mOP (similar to mAP)
                    if dtOrigin[tind, dind] <= self.OriginThres:
                        dtOriginThres[tind, dind] = 1
                    else:
                        dtOriginThres[tind, dind] = 0

                    # Calculate the difference between the pred and gt axis (degree)
                    dtAxis[tind, dind] = dot(gt_axis, pred_axis) / (
                        norm(gt_axis) * norm(pred_axis)
                    )
                    if dtAxis[tind, dind] < 0:
                        dtAxis[tind, dind] = -dtAxis[tind, dind]
                    dtAxis[tind, dind] = min(dtAxis[tind, dind], 1.0)
                    ## dtAxis used for evaluation metric MD
                    dtAxis[tind, dind] = np.arccos(dtAxis[tind, dind]) / np.pi * 180
                    ## dtAxisThres used for evaluation metric mDP
                    if dtAxis[tind, dind] <= self.AxisThres:
                        dtAxisThres[tind, dind] = 1
                    else:
                        dtAxisThres[tind, dind] = 0

        # set unmatched detections outside of area range to ignore
        a = np.array([d["area"] < aRng[0] or d["area"] > aRng[1] for d in dt]).reshape(
            (1, len(dt))
        )
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))
        # store results for given image and category
        result = {
            "image_id": imgId,
            "category_id": catId,
            "aRng": aRng,
            "maxDet": maxDet,
            "dtIds": [d["id"] for d in dt],
            "gtIds": [g["id"] for g in gt],
            "dtMatches": dtm,
            "gtMatches": gtm,
            "dtScores": [d["score"] for d in dt],
            "gtIgnore": gtIg,
            "dtIgnore": dtIg,
            "rotType": rotType,
            "gtType": gtType,
            # MotionNet
            "dtType": dtType,
            "dtOrigin": dtOrigin,
            "dtAxis": dtAxis,
            "dtOriginThres": dtOriginThres,
            "dtAxisThres": dtAxisThres,
        }

        if self.motionnet_type == "BMOC":
            result.update({"dtOriginWorld": dtOriginWorld, "dtAxisWorld": dtAxisWorld})
        if self.motionstate:
            result["dtState"] = dtState

        return result

    def accumulate(self, p=None):
        """
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        """
        print("Accumulating evaluation results...")
        tic = time.time()
        if not self.evalImgs:
            print("Please run evaluate() first")
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds) if p.useCats else 1
        A = len(p.areaRng)
        M = len(p.maxDets)
        precision = -np.ones(
            (T, R, K, A, M)
        )  # -1 for the precision of absent categories
        recall = -np.ones((T, K, A, M))
        scores = -np.ones((T, R, K, A, M))
        # MotionNet: Just calculate the motion scores on the TP cases
        type_scores = -np.ones((T, K, A, M))
        origin_scores = -np.ones((T, K, A, M))
        axis_scores = -np.ones((T, K, A, M))
        axis_rot_scores = -np.ones((T, K, A, M))
        axis_trans_scores = -np.ones((T, K, A, M))

        if self.motionnet_type == "BMOC":
            origin_world_scores = -np.ones((T, K, A, M))
            axis_world_scores = -np.ones((T, K, A, M))

        # MotionNet: calculate the mAP for type, originThres and axisThres
        if self.motionstate:
            mSP = -np.ones((T, R, K, A, M))
        mTP = -np.ones((T, R, K, A, M))
        # Below is for +MAO
        mMP = -np.ones((T, R, K, A, M))
        # Below is for +MA
        mTDP = -np.ones((T, R, K, A, M))
        mOP = -np.ones((T, R, K, A, M))
        mDP = -np.ones((T, R, K, A, M))
        mDP_rot = -np.ones((T, R, K, A, M))
        mDP_trans = -np.ones((T, R, K, A, M))

        # MotionNet: Record the number of instances used for each metric
        nIns_mAP_normal = -np.ones((T, K, A, M)) # for det, type, axis
        nIns_mAP_rot = -np.ones((T, K, A, M)) # for origin, axis_rot
        nIns_mAP_trans = -np.ones((T, K, A, M)) # for axis_trans

        nIns_precision_type = -np.ones((T, K, A, M)) # for type
        nIns_error_axis = -np.ones((T, K, A, M)) # for axis
        nIns_error_rot = -np.ones((T, K, A, M)) # for origin, axis_rot
        nIns_error_trans = -np.ones((T, K, A, M)) # for axis_trans

        nIns_total_pred = -np.ones((T, K, A, M)) # Tp + FP
        nIns_total_pred_rot = -np.ones((T, K, A, M)) # rotation prediction
        nIns_total_pred_trans = -np.ones((T, K, A, M)) # translation prediction

        total_origin_scores = -np.ones((T, K, A, M))
        total_axis_scores = -np.ones((T, K, A, M))
        total_axis_rot_scores = -np.ones((T, K, A, M))
        total_axis_trans_scores = -np.ones((T, K, A, M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [
            n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA
        ]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e["dtScores"][0:maxDet] for e in E])
                
                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind="mergesort")
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate(
                        [e["dtMatches"][:, 0:maxDet] for e in E], axis=1
                    )[:, inds]
                    dtIg = np.concatenate(
                        [e["dtIgnore"][:, 0:maxDet] for e in E], axis=1
                    )[:, inds]
                    gtIg = np.concatenate([e["gtIgnore"] for e in E])
                    gtType = np.concatenate([e["gtType"] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    nrot = np.sum((gtType == MOTION_TYPE["rotation"]) & (gtIg == 0))
                    ntrans = np.sum((gtType == MOTION_TYPE["translation"]) & (gtIg == 0))
                    if npig == 0:
                        continue
                    # MotionNet
                    if self.motionstate:
                        dtState = np.concatenate(
                            [e["dtState"][:, 0:maxDet] for e in E], axis=1
                        )[:, inds]
                    dtType = np.concatenate(
                        [e["dtType"][:, 0:maxDet] for e in E], axis=1
                    )[:, inds]
                    dtOrigin = np.concatenate(
                        [e["dtOrigin"][:, 0:maxDet] for e in E], axis=1
                    )[:, inds]
                    rotType = np.concatenate(
                        [e["rotType"][:, 0:maxDet] for e in E], axis=1
                    )[:, inds]

                    dtAxis = np.concatenate(
                        [e["dtAxis"][:, 0:maxDet] for e in E], axis=1
                    )[:, inds]
                    dtOriginThres = np.concatenate(
                        [e["dtOriginThres"][:, 0:maxDet] for e in E], axis=1
                    )[:, inds]
                    dtAxisThres = np.concatenate(
                        [e["dtAxisThres"][:, 0:maxDet] for e in E], axis=1
                    )[:, inds]

                    # Get the accumulated average score of dtType, dtOrigin and dtAxis
                    dtType_sum = np.zeros(np.shape(dtType))
                    dtOrigin_sum = np.zeros(np.shape(dtOrigin))
                    dtAxis_sum = np.zeros(np.shape(dtAxis))
                    dtAxis_rot_sum = np.zeros(np.shape(dtAxis))
                    dtAxis_trans_sum = np.zeros(np.shape(dtAxis))

                    for t in range(np.shape(dtType)[0]):
                        valid_type_number = 0
                        valid_axis_number = 0
                        valid_rot_number = 0
                        valid_trans_number = 0

                        for d in range(np.shape(dtType)[1]):
                            if d == 0:
                                if dtm[t, d] != 0 and dtIg[t, d] == 0:
                                    valid_type_number += 1
                                    dtType_sum[t, d] = dtType[t, d]
                                    if dtType[t, d] == 1:
                                        valid_axis_number += 1
                                        if rotType[t, d] == MOTION_TYPE["rotation"]:
                                            valid_rot_number += 1
                                            dtOrigin_sum[t, d] = dtOrigin[t, d]
                                            dtAxis_rot_sum[t, d] = dtAxis[t, d]
                                        elif (
                                            rotType[t, d] == MOTION_TYPE["translation"]
                                        ):
                                            valid_trans_number += 1
                                            dtAxis_trans_sum[t, d] = dtAxis[t, d]
                                        dtAxis_sum[t, d] = dtAxis[t, d]
                            else:
                                if dtm[t, d] != 0 and dtIg[t, d] == 0:
                                    valid_type_number += 1
                                    dtType_sum[t, d] = (
                                        dtType_sum[t, d - 1]
                                        + (dtType[t, d] - dtType_sum[t, d - 1])
                                        / valid_type_number
                                    )
                                    if dtType[t, d] == 1:
                                        valid_axis_number += 1
                                        dtAxis_sum[t, d] = (
                                            dtAxis_sum[t, d - 1]
                                            + (dtAxis[t, d] - dtAxis_sum[t, d - 1])
                                            / valid_axis_number
                                        )
                                        if rotType[t, d] == MOTION_TYPE["rotation"]:
                                            valid_rot_number += 1
                                            dtOrigin_sum[t, d] = (
                                                dtOrigin_sum[t, d - 1]
                                                + (
                                                    dtOrigin[t, d]
                                                    - dtOrigin_sum[t, d - 1]
                                                )
                                                / valid_rot_number
                                            )
                                            dtAxis_rot_sum[t, d] = (
                                                dtAxis_rot_sum[t, d - 1]
                                                + (
                                                    dtAxis[t, d]
                                                    - dtAxis_rot_sum[t, d - 1]
                                                )
                                                / valid_rot_number
                                            )
                                            dtAxis_trans_sum[t, d] = dtAxis_trans_sum[
                                                t, d - 1
                                            ]
                                        elif (
                                            rotType[t, d] == MOTION_TYPE["translation"]
                                        ):
                                            valid_trans_number += 1
                                            dtOrigin_sum[t, d] = dtOrigin_sum[t, d - 1]
                                            dtAxis_trans_sum[t, d] = (
                                                dtAxis_trans_sum[t, d - 1]
                                                + (
                                                    dtAxis[t, d]
                                                    - dtAxis_trans_sum[t, d - 1]
                                                )
                                                / valid_trans_number
                                            )
                                            dtAxis_rot_sum[t, d] = dtAxis_rot_sum[
                                                t, d - 1
                                            ]
                                    else:
                                        dtOrigin_sum[t, d] = dtOrigin_sum[t, d - 1]
                                        dtAxis_sum[t, d] = dtAxis_sum[t, d - 1]
                                        dtAxis_rot_sum[t, d] = dtAxis_rot_sum[t, d - 1]
                                        dtAxis_trans_sum[t, d] = dtAxis_trans_sum[
                                            t, d - 1
                                        ]

                                else:
                                    dtType_sum[t, d] = dtType_sum[t, d - 1]
                                    dtOrigin_sum[t, d] = dtOrigin_sum[t, d - 1]
                                    dtAxis_sum[t, d] = dtAxis_sum[t, d - 1]
                                    dtAxis_rot_sum[t, d] = dtAxis_rot_sum[t, d - 1]
                                    dtAxis_trans_sum[t, d] = dtAxis_trans_sum[t, d - 1]

                        if valid_type_number == 0:
                            nIns_precision_type[t, k, a, m] = -1
                        else:
                            nIns_precision_type[t, k, a, m] = valid_type_number

                        if valid_axis_number == 0:
                            nIns_error_axis[t, k, a, m] = -1
                        else:
                            nIns_error_axis[t, k, a, m] = valid_axis_number
                        
                        if valid_rot_number == 0:
                            nIns_error_rot[t, k, a, m] = -1
                        else:
                            nIns_error_rot[t, k, a, m] = valid_rot_number

                        if valid_trans_number == 0:
                            nIns_error_trans[t, k, a, m] = -1
                        else:
                            nIns_error_trans[t, k, a, m] = valid_trans_number

                        try:
                            if valid_type_number == 0:
                                dtType_sum[t, np.shape(dtType)[1] - 1] = -1
                            if valid_axis_number == 0:
                                dtAxis_sum[t, np.shape(dtType)[1] - 1] = -1
                            if valid_rot_number == 0:
                                dtOrigin_sum[t, np.shape(dtType)[1] - 1] = -1
                                dtAxis_rot_sum[t, np.shape(dtType)[1] - 1] = -1
                            if valid_trans_number == 0:
                                dtAxis_trans_sum[t, np.shape(dtType)[1] - 1] = -1
                        except:
                            # When training, at the start, some category may be missing, which may cause problems
                            pass
                    # import pdb
                    # pdb.set_trace()
                    # Process the dtOriginThres and dtAxisThres and dtType
                    rot_index = np.where(rotType[0] == MOTION_TYPE["rotation"])[0]
                    trans_index = np.where(rotType[0] == MOTION_TYPE["translation"])[0]

                    # For all translation, we set the origin to be true (because we don't really care about the origin in translation case)
                    processOriginThres = np.array(dtOriginThres)
                    processOriginThres[:, trans_index] = 1
                    dtMotionThres_tps = np.logical_and(
                        np.logical_and.reduce((dtm, dtType, dtAxisThres, processOriginThres)), np.logical_not(dtIg)
                    )
                    dtMotionThres_fps = np.logical_and(
                        np.logical_or.reduce((
                            np.logical_not(dtm), np.logical_not(dtType), np.logical_not(dtAxisThres), np.logical_not(processOriginThres)
                        )),
                        np.logical_not(dtIg),
                    )
                    dtMotionThres_tp_sum = np.cumsum(dtMotionThres_tps, axis=1).astype(
                        dtype=np.float
                    )
                    dtMotionThres_fp_sum = np.cumsum(dtMotionThres_fps, axis=1).astype(
                        dtype=np.float
                    )

                    dtTypeAxisThres_tps = np.logical_and(
                        np.logical_and.reduce((dtm, dtType, dtAxisThres)), np.logical_not(dtIg)
                    )
                    dtTypeAxisThres_fps = np.logical_and(
                        np.logical_or.reduce((
                            np.logical_not(dtm), np.logical_not(dtType), np.logical_not(dtAxisThres)
                        )),
                        np.logical_not(dtIg),
                    )
                    dtTypeAxisThres_tp_sum = np.cumsum(dtTypeAxisThres_tps, axis=1).astype(
                        dtype=np.float
                    )
                    dtTypeAxisThres_fp_sum = np.cumsum(dtTypeAxisThres_fps, axis=1).astype(
                        dtype=np.float
                    )

                    dtOriginThres_tps = np.logical_and(
                        np.logical_and(dtm, dtOriginThres), np.logical_not(dtIg)
                    )
                    dtOriginThres_fps = np.logical_and(
                        np.logical_or(
                            np.logical_not(dtm), np.logical_not(dtOriginThres)
                        ),
                        np.logical_not(dtIg),
                    )
                    dtOriginThres_tp_sum = np.cumsum(
                        dtOriginThres_tps[:, rot_index], axis=1
                    ).astype(dtype=np.float)
                    dtOriginThres_fp_sum = np.cumsum(
                        dtOriginThres_fps[:, rot_index], axis=1
                    ).astype(dtype=np.float)

                    dtAxisThres_tps = np.logical_and(
                        np.logical_and(dtm, dtAxisThres), np.logical_not(dtIg)
                    )
                    dtAxisThres_fps = np.logical_and(
                        np.logical_or(np.logical_not(dtm), np.logical_not(dtAxisThres)),
                        np.logical_not(dtIg),
                    )
                    dtAxisThres_tp_sum = np.cumsum(dtAxisThres_tps, axis=1).astype(
                        dtype=np.float
                    )
                    dtAxisThres_fp_sum = np.cumsum(dtAxisThres_fps, axis=1).astype(
                        dtype=np.float
                    )
                    dtAxisThres_rot_tp_sum = np.cumsum(
                        dtAxisThres_tps[:, rot_index], axis=1
                    ).astype(dtype=np.float)
                    dtAxisThres_rot_fp_sum = np.cumsum(
                        dtAxisThres_fps[:, rot_index], axis=1
                    ).astype(dtype=np.float)
                    dtAxisThres_trans_tp_sum = np.cumsum(
                        dtAxisThres_tps[:, trans_index], axis=1
                    ).astype(dtype=np.float)
                    dtAxisThres_trans_fp_sum = np.cumsum(
                        dtAxisThres_fps[:, trans_index], axis=1
                    ).astype(dtype=np.float)

                    dtType_tps = np.logical_and(
                        np.logical_and(dtm, dtType), np.logical_not(dtIg)
                    )
                    dtType_fps = np.logical_and(
                        np.logical_or(np.logical_not(dtm), np.logical_not(dtType)),
                        np.logical_not(dtIg),
                    )
                    dtType_tp_sum = np.cumsum(dtType_tps, axis=1).astype(dtype=np.float)
                    dtType_fp_sum = np.cumsum(dtType_fps, axis=1).astype(dtype=np.float)

                    if self.motionstate:
                        dtState_tps = np.logical_and(
                            np.logical_and(dtm, dtState), np.logical_not(dtIg)
                        )
                        dtState_fps = np.logical_and(
                            np.logical_or(np.logical_not(dtm), np.logical_not(dtState)),
                            np.logical_not(dtIg),
                        )
                        dtState_tp_sum = np.cumsum(dtState_tps, axis=1).astype(dtype=np.float)
                        dtState_fp_sum = np.cumsum(dtState_fps, axis=1).astype(dtype=np.float)

                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))
                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)

                    if (rot_index.shape[0] + trans_index.shape[0]) != 0:
                        nIns_total_pred[:, k, a, m] = rot_index.shape[0] + trans_index.shape[0]
                        matched_number = nIns_error_axis[:, k, a, m]
                        matched_number[np.where(matched_number == -1)] = 0
                        total_axis_scores[:, k, a, m] = (dtAxis_sum[:, -1] * matched_number + 90.0 * (nIns_total_pred[:, k, a, m] - matched_number)) / nIns_total_pred[:, k, a, m]
                        if rot_index.shape[0] != 0:
                            # There are rotation predictions
                            nIns_total_pred_rot[:, k, a, m] = rot_index.shape[0]
                            matched_rot_number = nIns_error_rot[:, k, a, m]
                            matched_rot_number[np.where(matched_rot_number == -1)] = 0
                            total_origin_scores[:, k, a, m] = (dtOrigin_sum[:, -1] * matched_rot_number + 1.0 * (nIns_total_pred_rot[:, k, a, m] - matched_rot_number)) / nIns_total_pred_rot[:, k, a, m]
                            total_axis_rot_scores[:, k, a, m] = (dtAxis_rot_sum[:, -1] * matched_rot_number + 90.0 * (nIns_total_pred_rot[:, k, a, m] - matched_rot_number)) / nIns_total_pred_rot[:, k, a, m]
                        if trans_index.shape[0] != 0:
                            # There are translation predictions
                            nIns_total_pred_trans[:, k, a, m] = trans_index.shape[0]
                            matched_trans_number = nIns_error_trans[:, k, a, m]
                            matched_trans_number[np.where(matched_trans_number == -1)] = 0
                            total_axis_trans_scores[:, k, a, m] = (dtAxis_trans_sum[:, -1] * matched_trans_number + 90.0 * (nIns_total_pred_trans[:, k, a, m] - matched_trans_number)) / nIns_total_pred_trans[:, k, a, m]

                    if self.motionnet_type == "BMOC":
                        dtOriginWorld = np.concatenate(
                            [e["dtOriginWorld"][:, 0:maxDet] for e in E], axis=1
                        )[:, inds]
                        dtAxisWorld = np.concatenate(
                            [e["dtAxisWorld"][:, 0:maxDet] for e in E], axis=1
                        )[:, inds]

                        dtOriginWorld_sum = np.zeros(np.shape(dtOriginWorld))
                        dtAxisWorld_sum = np.zeros(np.shape(dtAxisWorld))

                        for t in range(np.shape(dtOriginWorld)[0]):
                            valid_axis_number = 0
                            valid_rot_number = 0

                            for d in range(np.shape(dtOriginWorld)[1]):
                                if d == 0:
                                    if dtm[t, d] != 0 and dtIg[t, d] == 0:
                                        if dtType[t, d] == 1:
                                            valid_axis_number += 1
                                            dtAxisWorld_sum[t, d] = dtAxisWorld[t, d]
                                            if rotType[t, d] == MOTION_TYPE["rotation"]:
                                                valid_rot_number += 1
                                                dtOriginWorld_sum[t, d] = dtOriginWorld[
                                                    t, d
                                                ]
                                else:
                                    if dtm[t, d] != 0 and dtIg[t, d] == 0:
                                        if dtType[t, d] == 1:
                                            valid_axis_number += 1
                                            dtAxisWorld_sum[t, d] = (
                                                dtAxisWorld_sum[t, d - 1]
                                                + (
                                                    dtAxisWorld[t, d]
                                                    - dtAxisWorld_sum[t, d - 1]
                                                )
                                                / valid_axis_number
                                            )
                                            if rotType[t, d] == MOTION_TYPE["rotation"]:
                                                valid_rot_number += 1
                                                dtOriginWorld_sum[t, d] = (
                                                    dtOriginWorld_sum[t, d - 1]
                                                    + (
                                                        dtOriginWorld[t, d]
                                                        - dtOriginWorld_sum[t, d - 1]
                                                    )
                                                    / valid_rot_number
                                                )
                                            else:
                                                dtOriginWorld_sum[
                                                    t, d
                                                ] = dtOriginWorld_sum[t, d - 1]
                                        else:
                                            dtOriginWorld_sum[t, d] = dtOriginWorld_sum[
                                                t, d - 1
                                            ]
                                            dtAxisWorld_sum[t, d] = dtAxisWorld_sum[
                                                t, d - 1
                                            ]
                                    else:
                                        dtOriginWorld_sum[t, d] = dtOriginWorld_sum[
                                            t, d - 1
                                        ]
                                        dtAxisWorld_sum[t, d] = dtAxisWorld_sum[
                                            t, d - 1
                                        ]
                            try:
                                if valid_axis_number == 0:
                                    dtAxisWorld_sum[t, np.shape(dtType)[1] - 1] = -1
                                if valid_rot_number == 0:
                                    dtOriginWorld_sum[t, np.shape(dtType)[1] - 1] = -1
                            except:
                                pass
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        q = np.zeros((R,))
                        ss = np.zeros((R,))
                        # MotionNet: mTP, mOP, mDP
                        mMP_q = np.zeros((R,))
                        mTDP_q = np.zeros((R,))
                        mOP_q = np.zeros((R,))
                        mDP_q = np.zeros((R,))
                        mDP_rot_q = np.zeros((R,))
                        mDP_trans_q = np.zeros((R,))
                        mTP_q = np.zeros((R,))
                        if self.motionstate:
                            mSP_q = np.zeros((R,))

                        # MotionNet： MT, MO, MD
                        try:
                            type_scores[t, k, a, m] = dtType_sum[t, -1]
                            origin_scores[t, k, a, m] = dtOrigin_sum[t, -1]
                            axis_scores[t, k, a, m] = dtAxis_sum[t, -1]
                            axis_rot_scores[t, k, a, m] = dtAxis_rot_sum[t, -1]
                            axis_trans_scores[t, k, a, m] = dtAxis_trans_sum[t, -1]

                            if self.motionnet_type == "BMOC":
                                origin_world_scores[t, k, a, m] = dtOriginWorld_sum[
                                    t, -1
                                ]
                                axis_world_scores[t, k, a, m] = dtAxisWorld_sum[t, -1]
                        except:
                            pass

                        # MotionNet: mTP, mOP, mDP
                        dtMotionThres_tp = dtMotionThres_tp_sum[t]
                        dtMotionThres_fp = dtMotionThres_fp_sum[t]
                        dtMotionThres_pr = dtMotionThres_tp / (
                            dtMotionThres_tp + dtMotionThres_fp + np.spacing(1)
                        )

                        dtTypeAxisThres_tp = dtTypeAxisThres_tp_sum[t]
                        dtTypeAxisThres_fp = dtTypeAxisThres_fp_sum[t]
                        dtTypeAxisThres_pr = dtTypeAxisThres_tp / (
                            dtTypeAxisThres_tp + dtTypeAxisThres_fp + np.spacing(1)
                        )
                        

                        dtOriginThres_tp = dtOriginThres_tp_sum[t]
                        dtOriginThres_fp = dtOriginThres_fp_sum[t]
                        dtOriginThres_pr = dtOriginThres_tp / (
                            dtOriginThres_tp + dtOriginThres_fp + np.spacing(1)
                        )

                        dtAxisThres_tp = dtAxisThres_tp_sum[t]
                        dtAxisThres_fp = dtAxisThres_fp_sum[t]
                        dtAxisThres_pr = dtAxisThres_tp / (
                            dtAxisThres_tp + dtAxisThres_fp + np.spacing(1)
                        )

                        dtAxisThres_rot_tp = dtAxisThres_rot_tp_sum[t]
                        dtAxisThres_rot_fp = dtAxisThres_rot_fp_sum[t]
                        dtAxisThres_rot_pr = dtAxisThres_rot_tp / (
                            dtAxisThres_rot_tp + dtAxisThres_rot_fp + np.spacing(1)
                        )

                        dtAxisThres_trans_tp = dtAxisThres_trans_tp_sum[t]
                        dtAxisThres_trans_fp = dtAxisThres_trans_fp_sum[t]
                        dtAxisThres_trans_pr = dtAxisThres_trans_tp / (
                            dtAxisThres_trans_tp + dtAxisThres_trans_fp + np.spacing(1)
                        )

                        dtType_tp = dtType_tp_sum[t]
                        dtType_fp = dtType_fp_sum[t]
                        dtType_pr = dtType_tp / (dtType_tp + dtType_fp + np.spacing(1))

                        if self.motionstate:
                            dtState_tp = dtState_tp_sum[t]
                            dtState_fp = dtState_fp_sum[t]
                            dtState_pr = dtState_tp / (dtState_tp + dtState_fp + np.spacing(1))
                        
                        # Deal with only rotation mAP
                        if nrot != 0:
                            origin_rc = dtOriginThres_tp / nrot
                            axis_rot_rc = dtAxisThres_rot_tp / nrot
                        else:
                            origin_rc = -np.ones(np.shape(dtOriginThres_tp))
                            axis_rot_rc = -np.ones(np.shape(dtAxisThres_rot_tp))
                            mOP_q[:] = -1
                            mDP_rot_q[:] = -1
                        # Deal with only translation mAP
                        if ntrans != 0:
                            axis_trans_rc = dtAxisThres_trans_tp / ntrans
                        else:
                            axis_trans_rc = -np.ones(np.shape(dtAxisThres_trans_tp))
                            mDP_trans_q[:] = -1
                        # Deal with both cases
                        if self.motionstate:
                            state_rc = dtState_tp / npig
                        type_rc = dtType_tp / npig
                        axis_rc = dtAxisThres_tp / npig
                        motion_rc = dtMotionThres_tp / npig
                        typeaxis_rc = dtTypeAxisThres_tp / npig

                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist()
                        dtMotionThres_pr = dtMotionThres_pr.tolist()
                        dtTypeAxisThres_pr = dtTypeAxisThres_pr.tolist()
                        dtOriginThres_pr = dtOriginThres_pr.tolist()
                        dtAxisThres_pr = dtAxisThres_pr.tolist()
                        dtAxisThres_rot_pr = dtAxisThres_rot_pr.tolist()
                        dtAxisThres_trans_pr = dtAxisThres_trans_pr.tolist()
                        dtType_pr = dtType_pr.tolist()
                        if self.motionstate:
                            dtState_pr = dtState_pr.tolist()

                        q = q.tolist()
                        mOP_q = mOP_q.tolist()
                        mDP_rot_q = mDP_rot_q.tolist()
                        mDP_trans_q = mDP_trans_q.tolist()
                        mTP_q = mTP_q.tolist()
                        if self.motionstate:
                            mSP_q = mSP_q.tolist()

                        # Length for only rotation
                        for i in range(len(dtOriginThres_tp) - 1, 0, -1):
                            # MotionNet: smooth the mTP, mOP, mDP
                            if dtOriginThres_pr[i] > dtOriginThres_pr[i - 1]:
                                dtOriginThres_pr[i - 1] = dtOriginThres_pr[i]
                            if dtAxisThres_rot_pr[i] > dtAxisThres_rot_pr[i - 1]:
                                dtAxisThres_rot_pr[i - 1] = dtAxisThres_rot_pr[i]
                        # Length for only translation
                        for i in range(len(dtAxisThres_trans_tp) - 1, 0, -1):
                            if dtAxisThres_trans_pr[i] > dtAxisThres_trans_pr[i - 1]:
                                dtAxisThres_trans_pr[i - 1] = dtAxisThres_trans_pr[i]
                        # Length for both rotation and translation
                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]
                            if dtType_pr[i] > dtType_pr[i - 1]:
                                dtType_pr[i - 1] = dtType_pr[i]
                            if dtMotionThres_pr[i] > dtMotionThres_pr[i - 1]:
                                dtMotionThres_pr[i - 1] = dtMotionThres_pr[i]
                            if dtTypeAxisThres_pr[i] > dtTypeAxisThres_pr[i - 1]:
                                dtTypeAxisThres_pr[i - 1] = dtTypeAxisThres_pr[i]
                            if dtAxisThres_pr[i] > dtAxisThres_pr[i - 1]:
                                dtAxisThres_pr[i - 1] = dtAxisThres_pr[i]
                            if self.motionstate:
                                if dtState_pr[i] > dtState_pr[i - 1]:
                                    dtState_pr[i - 1] = dtState_pr[i]

                        origin_inds = np.searchsorted(origin_rc, p.recThrs, side="left")
                        try:
                            for ri, pi in enumerate(origin_inds):
                                mOP_q[ri] = dtOriginThres_pr[pi]
                        except:
                            pass

                        motion_inds = np.searchsorted(motion_rc, p.recThrs, side="left")
                        try:
                            for ri, pi in enumerate(motion_inds):
                                mMP_q[ri] = dtMotionThres_pr[pi]
                        except:
                            pass

                        typeaxis_inds = np.searchsorted(typeaxis_rc, p.recThrs, side="left")
                        try:
                            for ri, pi in enumerate(typeaxis_inds):
                                mTDP_q[ri] = dtTypeAxisThres_pr[pi]
                        except:
                            pass

                        axis_inds = np.searchsorted(axis_rc, p.recThrs, side="left")
                        try:
                            for ri, pi in enumerate(axis_inds):
                                mDP_q[ri] = dtAxisThres_pr[pi]
                        except:
                            pass

                        axis_rot_inds = np.searchsorted(
                            axis_rot_rc, p.recThrs, side="left"
                        )
                        try:
                            for ri, pi in enumerate(axis_rot_inds):
                                mDP_rot_q[ri] = dtAxisThres_rot_pr[pi]
                        except:
                            pass

                        axis_trans_inds = np.searchsorted(
                            axis_trans_rc, p.recThrs, side="left"
                        )
                        try:
                            for ri, pi in enumerate(axis_trans_inds):
                                mDP_trans_q[ri] = dtAxisThres_trans_pr[pi]
                        except:
                            pass

                        type_inds = np.searchsorted(type_rc, p.recThrs, side="left")
                        try:
                            for ri, pi in enumerate(type_inds):
                                mTP_q[ri] = dtType_pr[pi]
                        except:
                            pass

                        if self.motionstate:
                            state_inds = np.searchsorted(state_rc, p.recThrs, side="left")
                            try:
                                for ri, pi in enumerate(state_inds):
                                    mSP_q[ri] = dtState_pr[pi]
                            except:
                                pass

                        inds = np.searchsorted(rc, p.recThrs, side="left")
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                            
                        precision[t, :, k, a, m] = np.array(q)
                        scores[t, :, k, a, m] = np.array(ss)

                        # MotionNet: record mTP, mOP, mDP based on the recall of iou match
                        mMP[t, :, k, a, m] = np.array(mMP_q)
                        mTDP[t, :, k, a, m] = np.array(mTDP_q)
                        mOP[t, :, k, a, m] = np.array(mOP_q)
                        mDP[t, :, k, a, m] = np.array(mDP_q)
                        mDP_rot[t, :, k, a, m] = np.array(mDP_rot_q)
                        mDP_trans[t, :, k, a, m] = np.array(mDP_trans_q)
                        mTP[t, :, k, a, m] = np.array(mTP_q)
                        if self.motionstate:
                            mSP[t, :, k, a, m] = np.array(mSP_q)
                        # if (precision[t, :, k, a, m] == mDP_trans[t, :, k, a, m]).all() == False:
                        #     import pdb
                        #     pdb.set_trace()

                        try:
                            nIns_mAP_normal[t, k, a, m] = dtType_tp[-1] + dtType_fp[-1]
                            if nrot != 0:
                                nIns_mAP_rot[t, k, a, m] = dtAxisThres_rot_tp[-1] + dtAxisThres_rot_fp[-1]
                            else:
                                nIns_mAP_rot[t, k, a, m] = -1
                            if ntrans != 0:
                                nIns_mAP_trans[t, k, a, m] = dtAxisThres_trans_tp[-1] + dtAxisThres_trans_fp[-1]
                            else:
                                nIns_mAP_trans[t, k, a, m] = -1
                        except:
                            # During traing, there may be no gt
                            pass
        
        self.eval = {
            "params": p,
            "counts": [T, R, K, A, M],
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": precision,
            "recall": recall,
            "scores": scores,
            # MotionNet
            "type_scores": type_scores,
            "origin_scores": origin_scores,
            "axis_scores": axis_scores,
            "axis_rot_scores": axis_rot_scores,
            "axis_trans_scores": axis_trans_scores,
            "mMP": mMP,
            "mTDP": mTDP,
            "mOP": mOP,
            "mDP": mDP,
            "mDP_rot": mDP_rot,
            "mDP_trans": mDP_trans,
            "mTP": mTP,
            "nIns_mAP_normal": nIns_mAP_normal,
            "nIns_mAP_rot": nIns_mAP_rot,
            "nIns_mAP_trans": nIns_mAP_trans,
            "nIns_precision_type": nIns_precision_type,
            "nIns_error_axis": nIns_error_axis,
            "nIns_error_rot": nIns_error_rot,
            "nIns_error_trans": nIns_error_trans,
            "total_origin_scores": total_origin_scores,
            "total_axis_scores": total_axis_scores,
            "total_axis_rot_scores": total_axis_rot_scores,
            "total_axis_trans_scores": total_axis_trans_scores,
            "nIns_total_pred": nIns_total_pred,
            "nIns_total_pred_rot": nIns_total_pred_rot,
            "nIns_total_pred_trans": nIns_total_pred_trans,
        }

        if self.motionnet_type == "BMOC":
            self.eval.update(
                {
                    "origin_world_scores": origin_world_scores,
                    "axis_world_scores": axis_world_scores,
                }
            )

        if self.motionstate:
            self.eval["mSP"] = mSP

        toc = time.time()
        print("DONE (t={:0.2f}s).".format(toc - tic))

    def summarize(self):
        """
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        """

        def _summarize(ap=1, iouThr=None, areaRng="all", maxDets=100, cat=-1):
            p = self.params
            iStr = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"

            iouStr = (
                "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1])
                if iouThr is None
                else "{:0.2f}".format(iouThr)
            )

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                titleStr = "Average Precision (Detection)"
                typeStr = "(mAP_Detection)"
                # dimension of precision: [TxRxKxAxM]
                s = self.eval["precision"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, :, aind, mind]
                else:
                    s = s[:, :, cat, aind, mind]
            elif ap == 0:
                titleStr = "Average Recall (Detection)"
                typeStr = "(AR_Detection)"
                # dimension of recall: [TxKxAxM]
                s = self.eval["recall"]
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, aind, mind]
                else:
                    s = s[:, cat, aind, mind]
            # MotionNet: Motion Type (only on matched cases)
            elif ap == 2:
                titleStr = "Precision (Motion Type)"
                typeStr = "(Precision_Type)"
                # dimension of type_scores: [TxKxAxM]
                s = self.eval["type_scores"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, aind, mind]
                else:
                    s = s[:, cat, aind, mind]
            # MotionNet: Motion Origin (only on matched cases)
            elif ap == 3:
                titleStr = "Error (Motion Origin)"
                typeStr = "(ERR_Orig)"
                # dimension of origin_scores: [TxKxAxM]
                s = self.eval["origin_scores"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, aind, mind]
                else:
                    s = s[:, cat, aind, mind]
            # MotionNet: Motion Axis (only on matched cases)
            elif ap == 4:
                titleStr = "Error (Motion Axis Direction)"
                typeStr = "(ERR_ADir)"
                # dimension of axis_scores: [TxKxAxM]
                s = self.eval["axis_scores"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, aind, mind]
                else:
                    s = s[:, cat, aind, mind]
                # MotionNet: mOP similar to mAP
            elif ap == 5:
                titleStr = "Average Precision (Motion Origin Threshold)"
                typeStr = "(mAP_Orig)"
                # dimension of precision: [TxRxKxAxM]
                s = self.eval["mOP"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, :, aind, mind]
                else:
                    s = s[:, :, cat, aind, mind]
            # MotionNet: mDP similar to mAP
            elif ap == 6:
                titleStr = "Average Precision (Motion Axis Direction Threshold)"
                typeStr = "(mAP_ADir)"
                # dimension of precision: [TxRxKxAxM]
                s = self.eval["mDP"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, :, aind, mind]
                else:
                    s = s[:, :, cat, aind, mind]
            # MotionNet: mTP similar to mAP
            elif ap == 7:
                titleStr = "Average Precision (Motion Type Threshold)"
                typeStr = "(mAP_Type)"
                # dimension of precision: [TxRxKxAxM]s
                s = self.eval["mTP"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, :, aind, mind]
                else:
                    s = s[:, :, cat, aind, mind]
            # MotionNet: Motion Rotation Axis
            elif ap == 8:
                titleStr = "Average Rotation Axis Score"
                typeStr = "(ERR_ADir@r)"
                # dimension of axis_scores: [TxKxAxM]
                s = self.eval["axis_rot_scores"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, aind, mind]
                else:
                    s = s[:, cat, aind, mind]
            # MotionNet: Motion Translation Axis
            elif ap == 9:
                titleStr = "Average Translation Axis Score"
                typeStr = "(ERR_ADir@t)"
                # dimension of axis_scores: [TxKxAxM]
                s = self.eval["axis_trans_scores"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, aind, mind]
                else:
                    s = s[:, cat, aind, mind]
            # MotionNet: mDP_rot similar to mAP
            elif ap == 10:
                titleStr = (
                    "Average Precision (Motion Axis Direction Threshold rotation)"
                )
                typeStr = "(mAP_ADir@r)"
                # dimension of axis_scores: [TxRxKxAxM]
                s = self.eval["mDP_rot"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, :, aind, mind]
                else:
                    s = s[:, :, cat, aind, mind]
            # MotionNet: mDP_trans similar to mAP
            elif ap == 11:
                titleStr = (
                    "Average Precision (Motion Axis Direction Threshold translation)"
                )
                typeStr = "(mAP_ADir@t)"
                # dimension of axis_scores: [TxRxKxAxM]
                s = self.eval["mDP_trans"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, :, aind, mind]
                else:
                    s = s[:, :, cat, aind, mind]
            elif ap == 12:
                titleStr = "Average Pose Dimension Score "
                typeStr = "(Err_PDim)"
                # dimension of type_scores: [TxKxAxM]
                s = self.eval["pdim_scores"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, aind, mind]
                else:
                    s = s[:, cat, aind, mind]
            elif ap == 13:
                titleStr = "Average Pose Translation Score "
                typeStr = "(Err_PTrans)"
                # dimension of ptrans_scores: [TxKxAxM]
                s = self.eval["ptrans_scores"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, aind, mind]
                else:
                    s = s[:, cat, aind, mind]
            elif ap == 14:
                titleStr = "Average Pose Rotation Score "
                typeStr = "(Err_PRot)"
                # dimension of type_scores: [TxKxAxM]
                s = self.eval["prot_scores"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, aind, mind]
                else:
                    s = s[:, cat, aind, mind]
            elif ap == 15:
                titleStr = "Average World Origin Score "
                typeStr = "(Err_OrigWorld)"
                # dimension of type_scores: [TxKxAxM]
                s = self.eval["origin_world_scores"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, aind, mind]
                else:
                    s = s[:, cat, aind, mind]
            elif ap == 16:
                titleStr = "Average World Axis Score "
                typeStr = "(Err_ADirWorld)"
                # dimension of type_scores: [TxKxAxM]
                s = self.eval["axis_world_scores"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, aind, mind]
                else:
                    s = s[:, cat, aind, mind]
            elif ap == 17:
                titleStr = "Average Precision (Motion Pose Dimension Threshold)"
                typeStr = "(mPDP)"
                # dimension of type_scores: [TxKxAxM]
                s = self.eval["mPDP"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, :, aind, mind]
                else:
                    s = s[:, :, cat, aind, mind]
            elif ap == 18:
                titleStr = "Average Precision (Motion Pose Translation Threshold)"
                typeStr = "(mPTP)"
                # dimension of type_scores: [TxKxAxM]
                s = self.eval["mPTP"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, :, aind, mind]
                else:
                    s = s[:, :, cat, aind, mind]
            elif ap == 19:
                titleStr = "Average Precision (Motion Pose Rotation Threshold)"
                typeStr = "(mPRP)"
                # dimension of type_scores: [TxKxAxM]
                s = self.eval["mPRP"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, :, aind, mind]
                else:
                    s = s[:, :, cat, aind, mind]
            elif ap == 20:
                titleStr = "NumInstances for mAP (det, type, axis)"
                typeStr = "(nIns_mAP_normal)"
                # dimension of type_scores: [TxKxAxM]
                s = self.eval["nIns_mAP_normal"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, aind, mind]
                else:
                    s = s[:, cat, aind, mind]
            elif ap == 21:
                titleStr = "NumInstances for mAP (origin, axis_rot)"
                typeStr = "(nIns_mAP_rot)"
                # dimension of type_scores: [TxKxAxM]
                s = self.eval["nIns_mAP_rot"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, aind, mind]
                else:
                    s = s[:, cat, aind, mind]
            elif ap == 22:
                titleStr = "NumInstances for mAP (axis_trans)"
                typeStr = "(nIns_mAP_trans)"
                # dimension of type_scores: [TxKxAxM]
                s = self.eval["nIns_mAP_trans"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, aind, mind]
                else:
                    s = s[:, cat, aind, mind]
            elif ap == 23:
                titleStr = "NumInstances for precision (type)"
                typeStr = "(nIns_precision_type)"
                # dimension of type_scores: [TxKxAxM]
                s = self.eval["nIns_precision_type"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, aind, mind]
                else:
                    s = s[:, cat, aind, mind]
            elif ap == 24:
                titleStr = "NumInstances for error axis (axis)"
                typeStr = "(nIns_error_axis)"
                # dimension of type_scores: [TxKxAxM]
                s = self.eval["nIns_error_axis"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, aind, mind]
                else:
                    s = s[:, cat, aind, mind]
            elif ap == 25:
                titleStr = "NumInstances for error rot (origin, axis_rot)"
                typeStr = "(nIns_error_rot)"
                # dimension of type_scores: [TxKxAxM]
                s = self.eval["nIns_error_rot"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, aind, mind]
                else:
                    s = s[:, cat, aind, mind]
            elif ap == 26:
                titleStr = "NumInstances for error trans (axis_trans)"
                typeStr = "(nIns_error_trans)"
                # dimension of type_scores: [TxKxAxM]
                s = self.eval["nIns_error_trans"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, aind, mind]
                else:
                    s = s[:, cat, aind, mind]
            elif ap == 27:
                titleStr = "Error Mis(Motion Origin)"
                typeStr = "(ERR_mis_Orig)"
                # dimension of origin_scores: [TxKxAxM]
                s = self.eval["mis_origin_scores"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, aind, mind]
                else:
                    s = s[:, cat, aind, mind]
            # MotionNet: Motion Axis (only on matched cases)
            elif ap == 28:
                titleStr = "Error Mis(Motion Axis Direction)"
                typeStr = "(ERR_mis_ADir)"
                # dimension of axis_scores: [TxKxAxM]
                s = self.eval["mis_axis_scores"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, aind, mind]
                else:
                    s = s[:, cat, aind, mind]
            elif ap == 29:
                titleStr = "Average Rotation Axis Score Mis"
                typeStr = "(ERR_mis_ADir@r)"
                # dimension of axis_scores: [TxKxAxM]
                s = self.eval["mis_axis_rot_scores"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, aind, mind]
                else:
                    s = s[:, cat, aind, mind]
            # MotionNet: Motion Translation Axis
            elif ap == 30:
                titleStr = "Average Translation Axis Score Mis"
                typeStr = "(ERR_mis_ADir@t)"
                # dimension of axis_scores: [TxKxAxM]
                s = self.eval["mis_axis_trans_scores"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, aind, mind]
                else:
                    s = s[:, cat, aind, mind]
            elif ap == 31:
                titleStr = "Average Precision (All motion) Threshold)"
                typeStr = "(mAP_Motion)"
                # dimension of precision: [TxRxKxAxM]
                s = self.eval["mMP"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, :, aind, mind]
                else:
                    s = s[:, :, cat, aind, mind]
            # MotionNet: mTP similar to mAP
            elif ap == 32:
                titleStr = "Average Precision (Motion State)"
                typeStr = "(mAP_State)"
                # dimension of precision: [TxRxKxAxM]s
                s = self.eval["mSP"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, :, aind, mind]
                else:
                    s = s[:, :, cat, aind, mind]
            # MotionNet: motion type and axis mAP
            elif ap == 33:
                titleStr = "Average Precision (+MA) Threshold)"
                typeStr = "(mAP_+MA)"
                # dimension of precision: [TxRxKxAxM]
                s = self.eval["mTDP"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, :, aind, mind]
                else:
                    s = s[:, :, cat, aind, mind]
            elif ap == 34:
                titleStr = "total error of origin"
                typeStr = "(TotERR_Orig)"
                # dimension of precision: [TxRxKxAxM]
                s = self.eval["total_origin_scores"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, aind, mind]
                else:
                    s = s[:, cat, aind, mind]
            elif ap == 35:
                titleStr = "total error of axis"
                typeStr = "(TotERR_Adir)"
                # dimension of precision: [TxRxKxAxM]
                s = self.eval["total_axis_scores"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, aind, mind]
                else:
                    s = s[:, cat, aind, mind]
            elif ap == 36:
                titleStr = "total error of axis (rotation)"
                typeStr = "(TotERR_Adir@r)"
                # dimension of precision: [TxRxKxAxM]
                s = self.eval["total_axis_rot_scores"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, aind, mind]
                else:
                    s = s[:, cat, aind, mind]
            elif ap == 37:
                titleStr = "total error of axis (translation)"
                typeStr = "(TotERR_Adir@t)"
                # dimension of precision: [TxRxKxAxM]
                s = self.eval["total_axis_trans_scores"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, aind, mind]
                else:
                    s = s[:, cat, aind, mind]
            elif ap == 38:
                titleStr = "total number of predictions"
                typeStr = "(TotIns_pred)"
                # dimension of precision: [TxRxKxAxM]
                s = self.eval["nIns_total_pred"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, aind, mind]
                else:
                    s = s[:, cat, aind, mind]
            elif ap == 39:
                titleStr = "total number of predictions (rotation)"
                typeStr = "(TotIns_pred@r)"
                # dimension of precision: [TxRxKxAxM]
                s = self.eval["nIns_total_pred_rot"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, aind, mind]
                else:
                    s = s[:, cat, aind, mind]
            elif ap == 40:
                titleStr = "total number of predictions (translation)"
                typeStr = "(TotIns_pred@t)"
                # dimension of precision: [TxRxKxAxM]
                s = self.eval["nIns_total_pred_trans"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if cat == -1:
                    s = s[:, :, aind, mind]
                else:
                    s = s[:, cat, aind, mind]
            else:
                raise ValueError("Error in _summarize")

            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((7,))
            if self.PART_CAT == False:
                if self.MACRO_TYPE == False:
                    stats[0] = _summarize(1)
                    stats[1] = _summarize(1, iouThr=0.5, maxDets=self.params.maxDets[2])
                    stats[2] = _summarize(1, iouThr=0.75, maxDets=self.params.maxDets[2])
                    _summarize(1, iouThr=0.95, maxDets=self.params.maxDets[2])
                    _summarize(1, areaRng="small", maxDets=self.params.maxDets[2])
                    _summarize(1, areaRng="medium", maxDets=self.params.maxDets[2])
                    _summarize(1, areaRng="large", maxDets=self.params.maxDets[2])
                    _summarize(0, maxDets=self.params.maxDets[2])
                    _summarize(0, areaRng="small", maxDets=self.params.maxDets[2])
                    _summarize(0, areaRng="medium", maxDets=self.params.maxDets[2])
                    _summarize(0, areaRng="large", maxDets=self.params.maxDets[2])
                    # MotionNet: average motion threshold (different ious)
                    _summarize(31)
                    stats[3] = _summarize(31, iouThr=0.5, maxDets=self.params.maxDets[2])
                    _summarize(31, iouThr=0.75, maxDets=self.params.maxDets[2])
                    _summarize(31, iouThr=0.95, maxDets=self.params.maxDets[2])
                    # MotionNet: motion type (different ious)
                    _summarize(2)
                    _summarize(2, iouThr=0.5, maxDets=self.params.maxDets[2])
                    _summarize(2, iouThr=0.75, maxDets=self.params.maxDets[2])
                    _summarize(2, iouThr=0.95, maxDets=self.params.maxDets[2])
                else:
                    # MotionNet: average motion threshold (different ious)
                    _summarize(31)
                    _summarize(31, iouThr=0.5, maxDets=self.params.maxDets[2])
                    _summarize(31, iouThr=0.75, maxDets=self.params.maxDets[2])
                    _summarize(31, iouThr=0.95, maxDets=self.params.maxDets[2])

                # MotionNet: average MA threshold (different ious)
                _summarize(33)
                _summarize(33, iouThr=0.5, maxDets=self.params.maxDets[2])
                _summarize(33, iouThr=0.75, maxDets=self.params.maxDets[2])
                _summarize(33, iouThr=0.95, maxDets=self.params.maxDets[2])

                # MotionNet: average type threshold (different ious)
                _summarize(7)
                stats[4] = _summarize(7, iouThr=0.5, maxDets=self.params.maxDets[2])
                _summarize(7, iouThr=0.75, maxDets=self.params.maxDets[2])
                _summarize(7, iouThr=0.95, maxDets=self.params.maxDets[2])
                # MotionNet: motion origin (different ious)
                _summarize(3)
                _summarize(3, iouThr=0.5, maxDets=self.params.maxDets[2])
                _summarize(3, iouThr=0.75, maxDets=self.params.maxDets[2])
                _summarize(3, iouThr=0.95, maxDets=self.params.maxDets[2])
                # MotionNet: average origin threshold (different ious)
                _summarize(5)
                stats[5] = _summarize(5, iouThr=0.5, maxDets=self.params.maxDets[2])
                _summarize(5, iouThr=0.75, maxDets=self.params.maxDets[2])
                _summarize(5, iouThr=0.95, maxDets=self.params.maxDets[2])
                # MotionNet: motion axis (different ious)
                _summarize(4)
                _summarize(4, iouThr=0.5, maxDets=self.params.maxDets[2])
                _summarize(4, iouThr=0.75, maxDets=self.params.maxDets[2])
                _summarize(4, iouThr=0.95, maxDets=self.params.maxDets[2])
                # MotionNet: average axis direction threshold (different ious)
                _summarize(6)
                stats[6] = _summarize(6, iouThr=0.5, maxDets=self.params.maxDets[2])
                _summarize(6, iouThr=0.75, maxDets=self.params.maxDets[2])
                _summarize(6, iouThr=0.95, maxDets=self.params.maxDets[2])
                # MotionNet: average rotation axis error (different ious)
                _summarize(8)
                _summarize(8, iouThr=0.5, maxDets=self.params.maxDets[2])
                _summarize(8, iouThr=0.75, maxDets=self.params.maxDets[2])
                _summarize(8, iouThr=0.95, maxDets=self.params.maxDets[2])
                # MotionNet: average axis direction rotation threshold (different ious)
                _summarize(10)
                _summarize(10, iouThr=0.5, maxDets=self.params.maxDets[2])
                _summarize(10, iouThr=0.75, maxDets=self.params.maxDets[2])
                _summarize(10, iouThr=0.95, maxDets=self.params.maxDets[2])
                # MotionNet: average translation axis error (different ious)
                _summarize(9)
                _summarize(9, iouThr=0.5, maxDets=self.params.maxDets[2])
                _summarize(9, iouThr=0.75, maxDets=self.params.maxDets[2])
                _summarize(9, iouThr=0.95, maxDets=self.params.maxDets[2])
                # MotionNet: average axis direction translation threshold (different ious)
                _summarize(11)
                _summarize(11, iouThr=0.5, maxDets=self.params.maxDets[2])
                _summarize(11, iouThr=0.75, maxDets=self.params.maxDets[2])
                _summarize(11, iouThr=0.95, maxDets=self.params.maxDets[2])
                if self.motionstate:
                    # MotionNet: mAP state (different ious)
                    _summarize(32)
                    _summarize(32, iouThr=0.5, maxDets=self.params.maxDets[2])
                    _summarize(32, iouThr=0.75, maxDets=self.params.maxDets[2])
                    _summarize(32, iouThr=0.95, maxDets=self.params.maxDets[2])
                if self.motionnet_type == "BMOC":
                    # MotionNet: average origin world distance to the gt axis world
                    _summarize(15)
                    _summarize(15, iouThr=0.5, maxDets=self.params.maxDets[2])
                    _summarize(15, iouThr=0.75, maxDets=self.params.maxDets[2])
                    _summarize(15, iouThr=0.95, maxDets=self.params.maxDets[2])
                    # MotionNet: average degree between axis world and gt axis world
                    _summarize(16)
                    _summarize(16, iouThr=0.5, maxDets=self.params.maxDets[2])
                    _summarize(16, iouThr=0.75, maxDets=self.params.maxDets[2])
                    _summarize(16, iouThr=0.95, maxDets=self.params.maxDets[2])
                _summarize(20)
                _summarize(20, iouThr=0.5, maxDets=self.params.maxDets[2])
                _summarize(20, iouThr=0.75, maxDets=self.params.maxDets[2])
                _summarize(20, iouThr=0.95, maxDets=self.params.maxDets[2])
                _summarize(21)
                _summarize(21, iouThr=0.5, maxDets=self.params.maxDets[2])
                _summarize(21, iouThr=0.75, maxDets=self.params.maxDets[2])
                _summarize(21, iouThr=0.95, maxDets=self.params.maxDets[2])
                _summarize(22)
                _summarize(22, iouThr=0.5, maxDets=self.params.maxDets[2])
                _summarize(22, iouThr=0.75, maxDets=self.params.maxDets[2])
                _summarize(22, iouThr=0.95, maxDets=self.params.maxDets[2])
                _summarize(23)
                _summarize(23, iouThr=0.5, maxDets=self.params.maxDets[2])
                _summarize(23, iouThr=0.75, maxDets=self.params.maxDets[2])
                _summarize(23, iouThr=0.95, maxDets=self.params.maxDets[2])
                _summarize(24)
                _summarize(24, iouThr=0.5, maxDets=self.params.maxDets[2])
                _summarize(24, iouThr=0.75, maxDets=self.params.maxDets[2])
                _summarize(24, iouThr=0.95, maxDets=self.params.maxDets[2])
                _summarize(25)
                _summarize(25, iouThr=0.5, maxDets=self.params.maxDets[2])
                _summarize(25, iouThr=0.75, maxDets=self.params.maxDets[2])
                _summarize(25, iouThr=0.95, maxDets=self.params.maxDets[2])
                _summarize(26)
                _summarize(26, iouThr=0.5, maxDets=self.params.maxDets[2])
                _summarize(26, iouThr=0.75, maxDets=self.params.maxDets[2])
                _summarize(26, iouThr=0.95, maxDets=self.params.maxDets[2])

                _summarize(34)
                _summarize(34, iouThr=0.5, maxDets=self.params.maxDets[2])
                _summarize(34, iouThr=0.75, maxDets=self.params.maxDets[2])
                _summarize(34, iouThr=0.95, maxDets=self.params.maxDets[2])

                _summarize(35)
                _summarize(35, iouThr=0.5, maxDets=self.params.maxDets[2])
                _summarize(35, iouThr=0.75, maxDets=self.params.maxDets[2])
                _summarize(35, iouThr=0.95, maxDets=self.params.maxDets[2])

                _summarize(36)
                _summarize(36, iouThr=0.5, maxDets=self.params.maxDets[2])
                _summarize(36, iouThr=0.75, maxDets=self.params.maxDets[2])
                _summarize(36, iouThr=0.95, maxDets=self.params.maxDets[2])

                _summarize(37)
                _summarize(37, iouThr=0.5, maxDets=self.params.maxDets[2])
                _summarize(37, iouThr=0.75, maxDets=self.params.maxDets[2])
                _summarize(37, iouThr=0.95, maxDets=self.params.maxDets[2])

                _summarize(38)
                _summarize(38, iouThr=0.5, maxDets=self.params.maxDets[2])
                _summarize(38, iouThr=0.75, maxDets=self.params.maxDets[2])
                _summarize(38, iouThr=0.95, maxDets=self.params.maxDets[2])

                _summarize(39)
                _summarize(39, iouThr=0.5, maxDets=self.params.maxDets[2])
                _summarize(39, iouThr=0.75, maxDets=self.params.maxDets[2])
                _summarize(39, iouThr=0.95, maxDets=self.params.maxDets[2])

                _summarize(40)
                _summarize(40, iouThr=0.5, maxDets=self.params.maxDets[2])
                _summarize(40, iouThr=0.75, maxDets=self.params.maxDets[2])
                _summarize(40, iouThr=0.95, maxDets=self.params.maxDets[2])

            else:
                for cat in range(3):
                    if self.MACRO_TYPE == False:
                        _summarize(1, iouThr=0.5, maxDets=self.params.maxDets[2], cat=cat)
                        # MotionNet: motion type (different ious)
                        _summarize(2, iouThr=0.5, maxDets=self.params.maxDets[2], cat=cat)
                    # MotionNet: average motion threshold (different ious)
                    _summarize(31, iouThr=0.5, maxDets=self.params.maxDets[2], cat=cat)
                    _summarize(33, iouThr=0.5, maxDets=self.params.maxDets[2], cat=cat)
                    # MotionNet: average type threshold (different ious)
                    _summarize(7, iouThr=0.5, maxDets=self.params.maxDets[2], cat=cat)
                    # MotionNet: motion origin (different ious)
                    _summarize(3, iouThr=0.5, maxDets=self.params.maxDets[2], cat=cat)
                    # MotionNet: average origin threshold (different ious)
                    _summarize(5, iouThr=0.5, maxDets=self.params.maxDets[2], cat=cat)
                    _summarize(4, iouThr=0.5, maxDets=self.params.maxDets[2], cat=cat)
                    _summarize(6, iouThr=0.5, maxDets=self.params.maxDets[2], cat=cat)
                    # MotionNet: average rotation axis error (different ious)
                    _summarize(8, iouThr=0.5, maxDets=self.params.maxDets[2], cat=cat)
                    _summarize(10, iouThr=0.5, maxDets=self.params.maxDets[2], cat=cat)
                    _summarize(9, iouThr=0.5, maxDets=self.params.maxDets[2], cat=cat)
                    _summarize(11, iouThr=0.5, maxDets=self.params.maxDets[2], cat=cat)
                    if self.motionnet_type == "BMOC":
                        _summarize(15, iouThr=0.5, maxDets=self.params.maxDets[2], cat=cat)
                        _summarize(16, iouThr=0.5, maxDets=self.params.maxDets[2], cat=cat)
                    _summarize(20, iouThr=0.5, maxDets=self.params.maxDets[2], cat=cat)
                    _summarize(21, iouThr=0.5, maxDets=self.params.maxDets[2], cat=cat)
                    _summarize(22, iouThr=0.5, maxDets=self.params.maxDets[2], cat=cat)
                    _summarize(23, iouThr=0.5, maxDets=self.params.maxDets[2], cat=cat)
                    _summarize(24, iouThr=0.5, maxDets=self.params.maxDets[2], cat=cat)
                    _summarize(25, iouThr=0.5, maxDets=self.params.maxDets[2], cat=cat)
                    _summarize(26, iouThr=0.5, maxDets=self.params.maxDets[2], cat=cat)

            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=0.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=0.75)
            stats[3] = _summarize(1, maxDets=20, areaRng="medium")
            stats[4] = _summarize(1, maxDets=20, areaRng="large")
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=0.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=0.75)
            stats[8] = _summarize(0, maxDets=20, areaRng="medium")
            stats[9] = _summarize(0, maxDets=20, areaRng="large")
            return stats

        if not self.eval:
            raise Exception("Please run accumulate() first")
        iouType = self.params.iouType
        if iouType == "segm" or iouType == "bbox":
            summarize = _summarizeDets
        elif iouType == "keypoints":
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self):
        self.summarize()


class Params:
    """
    Params for coco evaluation api
    """

    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(
            0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
        )
        self.recThrs = np.linspace(
            0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
        )
        self.maxDets = [1, 10, 100]
        self.areaRng = [
            [0 ** 2, 1e5 ** 2],
            [0 ** 2, 32 ** 2],
            [32 ** 2, 96 ** 2],
            [96 ** 2, 1e5 ** 2],
        ]
        self.areaRngLbl = ["all", "small", "medium", "large"]
        self.useCats = 1

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(
            0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
        )
        self.recThrs = np.linspace(
            0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
        )
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ["all", "medium", "large"]
        self.useCats = 1
        self.kpt_oks_sigmas = (
            np.array(
                [
                    0.26,
                    0.25,
                    0.25,
                    0.35,
                    0.35,
                    0.79,
                    0.79,
                    0.72,
                    0.72,
                    0.62,
                    0.62,
                    1.07,
                    1.07,
                    0.87,
                    0.87,
                    0.89,
                    0.89,
                ]
            )
            / 10.0
        )

    def __init__(self, iouType="segm"):
        if iouType == "segm" or iouType == "bbox":
            self.setDetParams()
        elif iouType == "keypoints":
            self.setKpParams()
        else:
            raise Exception("iouType not supported")
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None
