from typing import Optional, Union, List

import cv2
import numpy as np

import mmcv
from mmcv.image import imread, imwrite
from mmcv.visualization.color import Color, color_val
from mmcv.visualization.image import ColorType, imshow
from mmtrack.models import MODELS, SiamRPN


@MODELS.register_module()
class SiameseRPN(SiamRPN):
    """Implement Siamese RPN which can be inited with pretrained init_cfg.
    NOTE The init_weights official implementation only init submodules.
    """

    def init_weights(self):
        if self.init_cfg is None:
            super().init_weights()
        else:
            super(SiamRPN, self).init_weights()

    def show_result(self,
                    img,
                    result,
                    gt=None,
                    color=None,
                    thickness=1,
                    show=False,
                    win_name='',
                    wait_time=0,
                    out_file=None,
                    **kwargs):
        """Visualize tracking results.

        Args:
            img (str or ndarray): The image to be displayed.
            result (dict): Tracking result.
                The value of key 'track_bboxes' is ndarray with shape (5, )
                in [tl_x, tl_y, br_x, br_y, score] format.
            gt (list): GT box. Default to None.
            color (str or tuple or Color, optional): color of bbox.
                Defaults to green.
            thickness (int, optional): Thickness of lines.
                Defaults to 1.
            show (bool, optional): Whether to show the image.
                Defaults to False.
            win_name (str, optional): The window name.
                Defaults to ''.
            wait_time (int, optional): Value of waitKey param.
                Defaults to 0.
            out_file (str, optional): The filename to write the image.
                Defaults to None.

        Returns:
            ndarray: Visualized image.
        """
        if color is None:
            color = ['green', 'red']

        assert isinstance(result, dict)
        track_bboxes = result.get('track_bboxes', None)
        assert track_bboxes.ndim == 1
        assert track_bboxes.shape[0] == 5
        track_bboxes = track_bboxes[:4]

        if gt is not None:
            assert isinstance(gt, list)
            gt_bboxes = gt[0].numpy()
        else:
            gt_bboxes = None

        imshow_bboxes(
            img,
            track_bboxes[np.newaxis, :],
            gt_bboxes=gt_bboxes,
            colors=color,
            thickness=thickness,
            show=show,
            win_name=win_name,
            wait_time=wait_time,
            out_file=out_file)
        return img


def imshow_bboxes(img: Union[str, np.ndarray],
                  pred_bboxes: Union[list, np.ndarray],
                  gt_bboxes: Union[list, np.ndarray],
                  colors: Union[List[ColorType], ColorType] = None,
                  top_k: int = -1,
                  thickness: int = 1,
                  show: bool = True,
                  win_name: str = '',
                  wait_time: int = 0,
                  out_file: Optional[str] = None):
    """Draw bboxes on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        pred_bboxes (list or ndarray): A list of ndarray of shape (k, 4).
        gt_bboxes (list or ndarray): A list of ndarray of shape (k, 4).
        colors (Color or str or tuple or int or ndarray): A list of colors.
        top_k (int): Plot the first k bboxes only if set positive.
        thickness (int): Thickness of lines.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str, optional): The filename to write the image.

    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    img = imread(img)
    img = np.ascontiguousarray(img)

    if isinstance(pred_bboxes, np.ndarray):
        pred_bboxes = [pred_bboxes]
    if isinstance(gt_bboxes, np.ndarray):
        gt_bboxes = [gt_bboxes]
    if colors is None:
        colors = ['green', 'red']
    if not isinstance(colors, list):
        colors = [colors for _ in range(len(pred_bboxes) + len(gt_bboxes))]
    colors = [color_val(c) for c in colors]
    assert len(pred_bboxes) + len(gt_bboxes) == len(colors)

    def _draw_box(_input_bboxes, _input_colors):
        for i, _bboxes in enumerate(_input_bboxes):
            _bboxes = _bboxes.astype(np.int32)
            if top_k <= 0:
                _top_k = _bboxes.shape[0]
            else:
                _top_k = min(top_k, _bboxes.shape[0])
            for j in range(_top_k):
                left_top = (_bboxes[j, 0], _bboxes[j, 1])
                right_bottom = (_bboxes[j, 2], _bboxes[j, 3])
                cv2.rectangle(
                    img, left_top, right_bottom,
                    _input_colors[i], thickness=thickness)

    _draw_box(pred_bboxes, colors)
    _draw_box(gt_bboxes, colors[len(pred_bboxes):])

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)
    return img