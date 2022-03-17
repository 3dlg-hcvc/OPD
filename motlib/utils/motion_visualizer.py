from fvcore.common.file_io import PathManager
from detectron2.utils.visualizer import (
    Visualizer,
    ColorMode,
    _create_text_labels,
    GenericMask,
)
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
)
from detectron2.utils.colormap import random_color

from PIL import Image
import numpy as np
from numpy.linalg import norm
import math

MOTION_TYPE = {0: "rotation", 1: "translation"}
_COLORS_CAT = {
    0: np.array([166, 206, 227]) / 255,
    1: np.array([31, 120, 180]) / 255,
    2: np.array([202, 178, 214]) / 255,
    3: np.array([106, 61, 154]) / 255,
    4: np.array([178, 223, 138]) / 255,
    5: np.array([51, 160, 44]) / 255,
}
_COLORS_LEVEL = {
    0: np.array([0, 255, 0]) / 255,
    1: np.array([255, 128, 0]) / 255,
    2: np.array([255, 0, 0]) / 255,
}


def getFocalLength(FOV, height, width=None):
    # FOV is in radius, should be vertical angle
    if width == None:
        f = height / (2 * math.tan(FOV / 2))
        return f
    else:
        fx = height / (2 * math.tan(FOV / 2))
        fy = fx / height * width
        return (fx, fy)


def camera_to_image(point, is_real=False, intrinsic_matrix=None):
    point_camera = np.array(point)
    # Calculate the camera intrinsic parameters (they are fixed in this project)
    if not is_real:
        # Below is for the MoionNet synthetic dataset intrinsic
        FOV = 50
        img_width = img_height = 256
        fx, fy = getFocalLength(FOV / 180 * math.pi, img_height, img_width)
        cy = img_height / 2
        cx = img_width / 2
        x = point_camera[0] * fx / (-point_camera[2]) + cx
        y = -(point_camera[1] * fy / (-point_camera[2])) + cy
    else:
        # Below is the for MotionREAL dataset 
        point_2d = np.dot(intrinsic_matrix, point_camera[:3])
        x = point_2d[0] / point_2d[2]
        y = point_2d[1] / point_2d[2]

    return (x, y)


def rotation_from_vectors(source, dest):
    a, b = (source / np.linalg.norm(source)).reshape(3), (
        dest / np.linalg.norm(dest)
    ).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rmat = np.eye(3) + kmat + np.matmul(kmat, kmat) * ((1 - c) / (s ** 2))
    return rmat


def rotatePoint(x, y, angle, scale):
    rad = np.pi * angle / 180
    x2 = np.cos(rad) * x - np.sin(rad) * y
    y2 = np.sin(rad) * x + np.cos(rad) * y
    return [x2 * scale, y2 * scale]


def circlePoints(axis, radius=0.5, num=50):
    angles = np.linspace(0, 2 * np.pi, num, endpoint=False)
    x_vec = np.cos(angles) * radius
    y_vec = np.sin(angles) * radius
    z_vec = np.zeros_like(x_vec) + 0.5
    points = np.stack((x_vec, y_vec, z_vec), axis=0)
    rot = rotation_from_vectors(np.array([0, 0, 1]), np.asarray(axis))
    points = np.matmul(rot, points)
    return points


def get_iou(bb1, bb2):
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[0] + bb1[2], bb2[0] + bb2[2])
    y_bottom = min(bb1[1] + bb1[3], bb2[1] + bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = bb1[2] * bb1[3]
    bb2_area = bb2[2] * bb2[3]
    iou = area / float(bb1_area + bb2_area - area)
    return iou


class MotionVisualizer(Visualizer):
    def draw_gt_instance(self, anno, part_id_json, is_real=False, intrinsic_matrix=None, line_length=1):
        # All annotations have been in the camera coordinate
        masks = [anno["segmentation"]]
        boxes = [BoxMode.convert(anno["bbox"], anno["bbox_mode"], BoxMode.XYXY_ABS)]
        labels = [anno["category_id"]]
        colors = None
        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get(
            "thing_colors"
        ):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]])
                for c in labels
            ]

        origins = [anno["motion"]["current_origin"]]
        # Calculate the 2d origin (Only consider draw only one origin)
        origins_4d = [origin[:] + [1] for origin in origins]
        origin_2d = [camera_to_image(origin, is_real, intrinsic_matrix) for origin in origins_4d]

        axises = [anno["motion"]["current_axis"]]
        new_point = list(np.array(origins[0]) + line_length * np.array(axises[0]))
        new_point = new_point[:] + [1]
        new_point = camera_to_image(new_point, is_real, intrinsic_matrix)

        arrow_p0 = rotatePoint(
            new_point[0] - origin_2d[0][0], new_point[1] - origin_2d[0][1], 30, 0.1
        )
        arrow_p1 = rotatePoint(
            new_point[0] - origin_2d[0][0], new_point[1] - origin_2d[0][1], -30, 0.1
        )
        circle_p = circlePoints(axises[0], 0.1, 50)
        circle_p = line_length * circle_p + np.repeat(
            np.asarray(origins[0])[:, np.newaxis], 50, axis=1
        )
        circle_p = circle_p.transpose()
        circle_p_2d = np.asarray([camera_to_image(p, is_real, intrinsic_matrix) for p in circle_p])

        self.draw_line(
            [origin_2d[0][0], new_point[0]],
            [origin_2d[0][1], new_point[1]],
            color=_COLORS_LEVEL[0],
            linewidth=2,
        )
        self.draw_line(
            [new_point[0] - arrow_p0[0], new_point[0]],
            [new_point[1] - arrow_p0[1], new_point[1]],
            color=_COLORS_LEVEL[0],
            linewidth=2,
        )
        self.draw_line(
            [new_point[0] - arrow_p1[0], new_point[0]],
            [new_point[1] - arrow_p1[1], new_point[1]],
            color=_COLORS_LEVEL[0],
            linewidth=2,
        )
        self.draw_polygon(
            circle_p_2d, color=_COLORS_LEVEL[0], edge_color=_COLORS_LEVEL[0], alpha=0.0
        )

        mtype = 0 if anno["motion"]["type"] == "rotation" else 1

        if not mtype:
            self.draw_circle(origin_2d[0], color=_COLORS_LEVEL[0], radius=5)

        names = self.metadata.get("thing_classes", None)
        if names:
            labels = [names[i] + "_" + anno["motion"]["type"] for i in labels]
        labels = [
            "{}".format(i) + ("|crowd" if a.get("iscrowd", 0) else "")
            for i, a in zip(labels, [anno])
        ]

        cat_id = anno["category_id"]
        self.overlay_instances(
            labels=labels,
            boxes=boxes,
            masks=masks,
            assigned_colors=[_COLORS_CAT[cat_id * 2 + mtype]],
        )

        part_id_json["partId"] = anno["motion"]["partId"]
        part_id_json["type"] = anno["motion"]["type"]
        part_id_json["category_id"] = anno["category_id"]

        return self.output

    def draw_prior(self, anno):
        # All annotations have been in the camera coordinate
        labels = [0]

        origin = anno["start"]
        origin_2d = anno["start_2d"]
        new_point = anno["end_2d"]

        axises = [anno["axises"]]
        print(axises)

        projection = anno["projMat"]

        arrow_p0 = rotatePoint(
            new_point[0] - origin_2d[0], new_point[1] - origin_2d[1], 30, 0.1
        )
        arrow_p1 = rotatePoint(
            new_point[0] - origin_2d[0], new_point[1] - origin_2d[1], -30, 0.1
        )

        circle_p = circlePoints(axises[0], 0.1, 50)
        circle_p = circle_p + np.repeat(np.asarray(origin)[:, np.newaxis], 50, axis=1)
        # circle_p = circle_p.transpose()
        circle_p = np.vstack((circle_p, np.ones(circle_p.shape[1])))
        circle_p_2d = np.dot(projection, circle_p)
        circle_p_2d = circle_p_2d / circle_p_2d[3, :]
        circle_p_2d = circle_p_2d[:2, :]
        circle_p_2d[0, :] = (circle_p_2d[0, :] + 1) / 2 * anno["img_size"]
        circle_p_2d[1, :] = (-circle_p_2d[1, :] + 1) / 2 * anno["img_size"]
        circle_p_2d = circle_p_2d.transpose()

        axis_diff = anno["error"]
        if axis_diff <= 2:
            axis_color = _COLORS_LEVEL[0]
        elif axis_diff > 2 and axis_diff <= 10:
            axis_color = _COLORS_LEVEL[1]
        elif axis_diff > 10:
            axis_color = _COLORS_LEVEL[2]

        print(axis_diff)

        self.draw_line(
            [origin_2d[0], new_point[0]],
            [origin_2d[1], new_point[1]],
            color=axis_color,
            linewidth=2,
        )
        self.draw_line(
            [new_point[0] - arrow_p0[0], new_point[0]],
            [new_point[1] - arrow_p0[1], new_point[1]],
            color=axis_color,
            linewidth=2,
        )
        self.draw_line(
            [new_point[0] - arrow_p1[0], new_point[0]],
            [new_point[1] - arrow_p1[1], new_point[1]],
            color=axis_color,
            linewidth=2,
        )
        self.draw_polygon(
            circle_p_2d, color=axis_color, edge_color=axis_color, alpha=0.0
        )

        mtype = 1

        if not mtype:
            self.draw_circle(origin_2d, color=_COLORS_LEVEL[0], radius=5)

        cat_id = 0
        labels = [
            "{}".format(i) + ("|crowd" if a.get("iscrowd", 0) else "")
            for i, a in zip(labels, [anno])
        ]
        # self.overlay_instances(
        #     labels=labels, boxes=None, masks=None, assigned_colors=[_COLORS_CAT[cat_id*2+mtype]]
        # )

        return self.output

    def draw_pred_instance(self, prediction, d, match, is_real=False, intrinsic_matrix=None, line_length=1, no_mask=False, diagonal_length=-1):
        if "annotations" in d:
            boxes = prediction.get("bbox", None)

            anno = None
            annos = d["annotations"]
            max_iou = -1
            if not len(annos):
                return None

            for gt_anno in annos:
                iou = get_iou(gt_anno["bbox"], boxes)
                if np.isnan(iou):
                    return False
                if iou > max_iou:
                    max_iou = iou
                    anno = gt_anno
        else:
            max_iou = -1
            boxes = prediction.get("bbox", None)
            anno = d
            boxes = prediction.get("bbox", None)
            iou = get_iou(anno["bbox"], boxes)
            if iou > max_iou:
                max_iou = iou

        boxes = [BoxMode.convert(boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)]

        # Based on the motion type, determine to visualize the predicted motion origin or gt motion origin
        # For translation joint, the motion origin is meaningless
        pred_type = prediction["mtype"]
        if pred_type == 1:
            pred_origin = anno["motion"]["current_origin"]
        else:
            pred_origin = prediction["morigin"]

        # Prepare the predicted origin and predicted axis
        pred_origin_4d = pred_origin + [1]
        pred_origin_2d = camera_to_image(pred_origin_4d, is_real, intrinsic_matrix)
        pred_axis = np.array(prediction["maxis"])
        pred_axis = list(pred_axis / norm(pred_axis))
        pred_new_point = list(np.array(pred_origin) + line_length * np.array(pred_axis))
        pred_new_point = pred_new_point + [1]
        pred_new_point = camera_to_image(pred_new_point, is_real, intrinsic_matrix)

        # Prepare the gt origin and gt axis
        gt_origin = anno["motion"]["current_origin"]
        gt_origin_4d = gt_origin + [1]
        gt_origin_2d = camera_to_image(gt_origin_4d, is_real, intrinsic_matrix)
        gt_axis = anno["motion"][
            "current_axis"
        ]  # gt_axis has been normalized in the annotation
        gt_new_point = list(np.array(gt_origin) + line_length * np.array(gt_axis))
        gt_new_point = gt_new_point + [1]
        gt_new_point = camera_to_image(gt_new_point, is_real, intrinsic_matrix)

        # Caluculate the axis and origin error to determine the color for the visualization of axis and origin
        axis_diff = (
            np.arccos(
                np.abs(
                    np.dot(np.array(gt_axis), np.array(pred_axis))
                    / (norm(pred_axis) * norm(gt_axis))
                )
            )
            / np.pi
            * 180.0
        )
        if axis_diff <= 5:
            axis_color = _COLORS_LEVEL[0]
        elif axis_diff > 5 and axis_diff <= 10:
            axis_color = _COLORS_LEVEL[1]
        elif axis_diff > 10:
            axis_color = _COLORS_LEVEL[2]

        if diagonal_length == -1:
            raise ValueError("diagonal length error")

        origin_diff = np.linalg.norm(
            np.cross(np.array(pred_origin) - np.array(gt_origin), np.array(gt_axis))
        ) / np.linalg.norm(gt_axis) / diagonal_length
        if origin_diff <= 0.1:
            origin_color = _COLORS_LEVEL[0]
        elif origin_diff > 0.1 and origin_diff <= 0.25:
            origin_color = _COLORS_LEVEL[1]
        elif origin_diff > 0.25:
            origin_color = _COLORS_LEVEL[2]

        # Visualize gt
        gt_color = np.array([0, 0, 255]) / 255
        gt_arrow_p0 = rotatePoint(
            gt_new_point[0] - gt_origin_2d[0],
            gt_new_point[1] - gt_origin_2d[1],
            30,
            0.1,
        )
        gt_arrow_p1 = rotatePoint(
            gt_new_point[0] - gt_origin_2d[0],
            gt_new_point[1] - gt_origin_2d[1],
            -30,
            0.1,
        )
        gt_circle_p = circlePoints(gt_axis, 0.1, 50)
        gt_circle_p = line_length * gt_circle_p + np.repeat(
            np.asarray(gt_origin)[:, np.newaxis], 50, axis=1
        )
        gt_circle_p = gt_circle_p.transpose()
        gt_circle_p_2d = np.asarray([camera_to_image(p, is_real, intrinsic_matrix) for p in gt_circle_p])
        self.draw_line(
            [gt_origin_2d[0], gt_new_point[0]],
            [gt_origin_2d[1], gt_new_point[1]],
            color=gt_color,
            linewidth=2,
        )
        self.draw_line(
            [gt_new_point[0] - gt_arrow_p0[0], gt_new_point[0]],
            [gt_new_point[1] - gt_arrow_p0[1], gt_new_point[1]],
            color=gt_color,
            linewidth=2,
        )
        self.draw_line(
            [gt_new_point[0] - gt_arrow_p1[0], gt_new_point[0]],
            [gt_new_point[1] - gt_arrow_p1[1], gt_new_point[1]],
            color=gt_color,
            linewidth=2,
        )
        self.draw_polygon(
            gt_circle_p_2d, color=gt_color, edge_color=gt_color, alpha=0.0
        )
        if pred_type == 0:
            # self.draw_text("origin_error: {:.3f}".format(origin_diff), (origin_2d[0][0], origin_2d[0][1]-10*text_y_offset), color="c")
            self.draw_circle(gt_origin_2d, color=gt_color, radius=5)

        # Visualize the predicted axis
        pred_arrow_p0 = rotatePoint(
            pred_new_point[0] - pred_origin_2d[0],
            pred_new_point[1] - pred_origin_2d[1],
            30,
            0.1,
        )
        pred_arrow_p1 = rotatePoint(
            pred_new_point[0] - pred_origin_2d[0],
            pred_new_point[1] - pred_origin_2d[1],
            -30,
            0.1,
        )
        pred_circle_p = circlePoints(pred_axis, 0.1, 50)
        pred_circle_p = line_length * pred_circle_p + np.repeat(
            np.asarray(pred_origin)[:, np.newaxis], 50, axis=1
        )
        pred_circle_p = pred_circle_p.transpose()
        pred_circle_p_2d = np.asarray([camera_to_image(p, is_real, intrinsic_matrix) for p in pred_circle_p])
        # text_y_offset = 1 if (new_point[1]-origin_2d[0][1]) > 0 else -1
        # self.draw_text("axis_error: {:.3f}".format(axis_diff), (origin_2d[0][0], origin_2d[0][1]-20*text_y_offset), color="tan")
        self.draw_line(
            [pred_origin_2d[0], pred_new_point[0]],
            [pred_origin_2d[1], pred_new_point[1]],
            color=axis_color,
            linewidth=2,
        )
        self.draw_line(
            [pred_new_point[0] - pred_arrow_p0[0], pred_new_point[0]],
            [pred_new_point[1] - pred_arrow_p0[1], pred_new_point[1]],
            color=axis_color,
            linewidth=2,
        )
        self.draw_line(
            [pred_new_point[0] - pred_arrow_p1[0], pred_new_point[0]],
            [pred_new_point[1] - pred_arrow_p1[1], pred_new_point[1]],
            color=axis_color,
            linewidth=2,
        )
        self.draw_polygon(
            pred_circle_p_2d, color=axis_color, edge_color=axis_color, alpha=0.0
        )
        if pred_type == 0:
            # self.draw_text("origin_error: {:.3f}".format(origin_diff), (origin_2d[0][0], origin_2d[0][1]-10*text_y_offset), color="c")
            self.draw_circle(pred_origin_2d, color=origin_color, radius=5)

        # Assign color to the segmentation
        cat_id = prediction.get("category_id", None)
        color_cat = _COLORS_CAT[cat_id * 2 + pred_type]

        scores = [prediction.get("score", None)]
        classes = [prediction.get("category_id", None)]
        labels = _create_text_labels_motion(
            classes,
            scores,
            self.metadata.get("thing_classes", None),
            MOTION_TYPE[pred_type],
        )
        keypoints = prediction.get("keypoints", None)
        if prediction.get("segmentation"):
            import pycocotools.mask as mask_util

            masks = [prediction.get("segmentation")]
        else:
            masks = None

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get(
            "thing_colors"
        ):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]])
                for c in classes
            ]
            alpha = 0.8
        else:
            colors = [color_cat]
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.img = self._create_grayscale_image(
                (mask_util.decode(prediction.get("segmentation")).any() > 0).numpy()
            )
            alpha = 0.3

        match["iou"] = max_iou
        # Add the gt information
        match["gt"] = {}
        match["gt"]["partId"] = anno["motion"]["partId"]
        match["gt"]["label"] = anno["motion"]["label"]
        match["gt"]["type"] = anno["motion"]["type"]
        match["gt"]["category_id"] = anno["category_id"]
        match["gt"]["origin"] = gt_origin
        match["gt"]["axis"] = gt_axis
        # add the prediction information
        match["pred"] = {}
        match["pred"]["score"] = scores[0]
        match["pred"]["type"] = pred_type
        match["pred"]["category_id"] = cat_id
        match["pred"]["origin"] = pred_origin
        match["pred"]["axis"] = pred_axis
        # add additional information
        match["axis_error"] = axis_diff
        match["origin_error"] = origin_diff
        match["match"] = (
            int(pred_type)
            == int(
                list(MOTION_TYPE.keys())[
                    list(MOTION_TYPE.values()).index(anno["motion"]["type"])
                ]
            )
        ) and (cat_id == anno["category_id"])

        if no_mask:
            masks = None

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output

    def draw_pred_only(self, prediction, prob):
        scores = prediction.scores if prediction.has("scores") else None
        if scores.numpy()[0] < prob:
            return None

        origins = list(prediction.morigin.numpy())
        origins = [list(origin) for origin in origins]

        axises = list(prediction.maxis.numpy())
        axises = [list(axis) for axis in axises]

        types = list(prediction.mtype.numpy())
        classes = prediction.pred_classes if prediction.has("pred_classes") else None

        color_cat = _COLORS_CAT[classes.numpy()[0] * 2 + types[0]]

        origins_4d = [origin[:] + [1] for origin in origins]
        origin_2d = [camera_to_image(origin) for origin in origins_4d]

        new_point = list(np.array(origins[0]) + np.array(axises[0]))
        new_point = new_point[:] + [1]
        new_point = camera_to_image(new_point)

        axis_color = _COLORS_LEVEL[0]
        origin_color = _COLORS_LEVEL[0]

        arrow_p0 = rotatePoint(
            new_point[0] - origin_2d[0][0], new_point[1] - origin_2d[0][1], 30, 0.1
        )
        arrow_p1 = rotatePoint(
            new_point[0] - origin_2d[0][0], new_point[1] - origin_2d[0][1], -30, 0.1
        )
        circle_p = circlePoints(axises[0], 0.1, 50)
        circle_p = circle_p + np.repeat(
            np.asarray(origins[0])[:, np.newaxis], 50, axis=1
        )
        circle_p = circle_p.transpose()
        circle_p_2d = np.asarray([camera_to_image(p) for p in circle_p])

        # text_y_offset = 1 if (new_point[1]-origin_2d[0][1]) > 0 else -1
        # self.draw_text("axis_error: {:.3f}".format(axis_diff), (origin_2d[0][0], origin_2d[0][1]-20*text_y_offset), color="tan")
        self.draw_line(
            [origin_2d[0][0], new_point[0]],
            [origin_2d[0][1], new_point[1]],
            color=axis_color,
            linewidth=2,
        )
        self.draw_line(
            [new_point[0] - arrow_p0[0], new_point[0]],
            [new_point[1] - arrow_p0[1], new_point[1]],
            color=axis_color,
            linewidth=2,
        )
        self.draw_line(
            [new_point[0] - arrow_p1[0], new_point[0]],
            [new_point[1] - arrow_p1[1], new_point[1]],
            color=axis_color,
            linewidth=2,
        )
        self.draw_polygon(
            circle_p_2d, color=axis_color, edge_color=axis_color, alpha=0.0
        )

        if types[0] == 0:
            # self.draw_text("origin_error: {:.3f}".format(origin_diff), (origin_2d[0][0], origin_2d[0][1]-10*text_y_offset), color="c")
            self.draw_circle(origin_2d[0], color=origin_color, radius=5)

        boxes = prediction.pred_boxes if prediction.has("pred_boxes") else None
        labels = _create_text_labels_motion(
            classes,
            scores,
            self.metadata.get("thing_classes", None),
            MOTION_TYPE[types[0]],
        )
        keypoints = (
            prediction.pred_keypoints if prediction.has("pred_keypoints") else None
        )

        if prediction.has("pred_masks"):
            masks = np.asarray(prediction.pred_masks)
            masks = [
                GenericMask(x, self.output.height, self.output.width) for x in masks
            ]
        else:
            masks = None

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get(
            "thing_colors"
        ):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]])
                for c in classes
            ]
            alpha = 0.8
        else:
            colors = [color_cat]
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.img = self._create_grayscale_image(
                (prediction.pred_masks.any(dim=0) > 0).numpy()
            )
            alpha = 0.3

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output


def _create_text_labels_motion(classes, scores, class_names, motion_type):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):

    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None and class_names is not None and len(class_names) > 1:
        labels = [class_names[i] for i in classes]
        labels = [label + "_" + motion_type for label in labels]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    return labels
