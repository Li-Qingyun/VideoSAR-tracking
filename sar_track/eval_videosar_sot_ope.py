# Modified from mmtrack/core/evaluation/eval_sot_ope.py
from typing import List, Optional, Tuple, Dict

import numpy as np
from mmtrack.core.evaluation.eval_sot_ope import success_error, bbox_overlaps

ArrayList = List[np.ndarray]


def eval_sot_ope(results: List[ArrayList],
                 annotations: ArrayList,
                 visible_infos: Optional[ArrayList] = None,
                 iou_th: Optional[np.ndarray] = None,
                 pixel_offset_th: Optional[np.ndarray] = None
                 ) -> Tuple[Dict, Dict]:
    """Evaluation in OPE protocol.

    Args:
        results (list[list[ndarray]]): The first list contains the tracking
            results of each video. The second list contains the tracking
            results of each frame in one video. The ndarray denotes the
            tracking box in [tl_x, tl_y, br_x, br_y] format.
        annotations (list[ndarray]): The list contains the bbox
            annotations of each video. The ndarray is gt_bboxes of one video.
            It's in (N, 4) shape. Each bbox is in (x1, y1, x2, y2) format.
        visible_infos (list[ndarray] | None): If not None, the list
            contains the visible information of each video. The ndarray is
            visibility (with bool type) of object in one video. It's in (N,)
            shape. Default to None.
        iou_th (ndarray | None): The `iou_th` of `success_overlap` for
            calculating `success`. If `None`, the `iou_th` will be
            `np.arange(0, 1.05, 0.05)`. Default to `None`.
        pixel_offset_th (ndarray | None): The `pixel_offset_th` of
            `success_error` for calculating `precision` and `normed precision`.
            If `None`, the `pixel_offset_th` will be `np.arange(0, 51, 1)`.
            Default to `None`.

    Returns:
        tuple[dict[str, float]]: OPE style evaluation metric (i.e. success,
        norm precision and precision).
    """
    success_results = []
    precision_results = []
    norm_precision_results = []
    if visible_infos is None:
        visible_infos = [np.array([True] * len(_)) for _ in annotations]
    for single_video_results, single_video_gt_bboxes, single_video_visible in zip(  # noqa
            results, annotations, visible_infos):
        pred_bboxes = np.stack(single_video_results)
        assert len(pred_bboxes) == len(single_video_gt_bboxes)
        video_length = len(single_video_results)

        gt_bboxes = single_video_gt_bboxes[single_video_visible]
        pred_bboxes = pred_bboxes[single_video_visible]

        # eval success based on iou
        if iou_th is None:
            iou_th = np.arange(0, 1.05, 0.05)
        success_results.append(
            success_overlap(gt_bboxes, pred_bboxes, iou_th, video_length))

        # eval precision
        gt_bboxes_center = np.array(
            (0.5 * (gt_bboxes[:, 2] + gt_bboxes[:, 0]),
             0.5 * (gt_bboxes[:, 3] + gt_bboxes[:, 1]))).T
        pred_bboxes_center = np.array(
            (0.5 * (pred_bboxes[:, 2] + pred_bboxes[:, 0]),
             0.5 * (pred_bboxes[:, 3] + pred_bboxes[:, 1]))).T
        if pixel_offset_th is None:
            pixel_offset_th = np.arange(0, 51, 1)
        precision_results.append(
            success_error(gt_bboxes_center, pred_bboxes_center,
                          pixel_offset_th, video_length))

        # eval normed precision
        gt_bboxes_wh = np.array((gt_bboxes[:, 2] - gt_bboxes[:, 0],
                                 gt_bboxes[:, 3] - gt_bboxes[:, 1])).T
        norm_gt_bboxes_center = gt_bboxes_center / (gt_bboxes_wh + 1e-16)
        norm_pred_bboxes_center = pred_bboxes_center / (gt_bboxes_wh + 1e-16)
        norm_pixel_offset_th = pixel_offset_th / 100.
        norm_precision_results.append(
            success_error(norm_gt_bboxes_center, norm_pred_bboxes_center,
                          norm_pixel_offset_th, video_length))

    success = np.mean(success_results) * 100
    precision = np.mean(precision_results, axis=0)[20] * 100
    norm_precision = np.mean(norm_precision_results, axis=0)[20] * 100
    eval_results = dict(
        success=success, norm_precision=norm_precision, precision=precision)
    meta_results = dict(success=dict(results=success_results,
                                     threshold=iou_th,
                                     threshold_name='iou_th'),
                        norm_precision=dict(results=norm_precision_results,
                                            threshold=norm_pixel_offset_th,
                                            threshold_name='norm_pixel_offset_th'),  # noqa
                        precision=dict(results=precision_results,
                                       threshold=pixel_offset_th,
                                       threshold_name='pixel_offset_th'))
    return eval_results, meta_results


def success_overlap(gt_bboxes, pred_bboxes, iou_th, video_length):
    """Modified from mmtrack/core/evaluation/eval_sot_ope.py
    Evaluation based on iou.

    Args:
        gt_bboxes (ndarray): of shape (video_length, 4) in
            [tl_x, tl_y, br_x, br_y] format.
        pred_bboxes (ndarray): of shape (video_length, 4) in
            [tl_x, tl_y, br_x, br_y] format.
        iou_th (ndarray): Different threshold of iou. Typically is set to
            `np.arange(0, 1.05, 0.05)`.
        video_length (int): Video length.

    Returns:
        ndarray: The evaluation results at different threshold of iou.
    """
    success = np.zeros(len(iou_th))
    iou = np.ones(len(gt_bboxes)) * (-1)
    valid = (gt_bboxes[:, 2] > gt_bboxes[:, 0]) & (
        gt_bboxes[:, 3] > gt_bboxes[:, 1])
    iou_matrix = bbox_overlaps(gt_bboxes[valid], pred_bboxes[valid])
    iou[valid] = iou_matrix[np.arange(len(gt_bboxes[valid])),
                            np.arange(len(gt_bboxes[valid]))]

    for i in range(len(iou_th)):
        success[i] = np.sum(iou > iou_th[i]) / float(video_length)
    return success