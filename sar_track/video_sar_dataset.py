import os
import time

import numpy as np
from mmcv import print_log
from mmtrack.datasets import DATASETS, BaseSOTDataset

from sar_track.eval_videosar_sot_ope import eval_sot_ope


@DATASETS.register_module()
class VideoSARDataset(BaseSOTDataset):
    """Video SAR interface of single object tracking.
    Modified from `UAV123Dataset
    <https://github.com/open-mmlab/mmtracking/blob/master/mmtrack/datasets/uav123_dataset.py>`_.  # noqa
    """

    selected_train_targets = [
        'left_target01', 'left_target02', 'left_target05', 'left_target06',
        'left_target08', 'left_target09', 'left_target10', 'left_target11',
        'left_target13', 'left_target15', 'left_target16', 'left_target17',
        'left_target18', 'left_target19', 'left_target21', 'left_target22',
        'left_target23', 'left_target25', 'left_target28', 'left_target29',
        'left_target31',
        'right_target01', 'right_target02', 'right_target04', 'right_target06',
        'right_target07', 'right_target09', 'right_target11', 'right_target12',
        'right_target14']

    def load_data_infos(self, split='all'):
        """Load dataset information.

        Args:
            split (str, optional): Dataset split, can be 'left', 'right',
                'train', 'test', and 'all'. Defaults to 'all'.

        Returns:
            list[dict]: The length of the list is the number of videos. The
                inner dict is in the following format:
                    {
                        'video_path': the video path
                        'ann_path': the annotation path
                        'start_frame_id': the starting frame number contained
                            in the image name
                        'end_frame_id': the ending frame number contained in
                            the image name
                        'framename_template': the template of image name
                    }
        """
        assert split in ('all', 'left', 'right', 'train', 'test')

        print('Loading Video SAR dataset...')
        start_time = time.time()
        data_infos = []
        data_infos_str = self.loadtxt(
            self.ann_file, return_array=False).split('\n')
        # the first line of annotation file is a dataset comment.
        for line in data_infos_str[1:]:  # The first line is tips message.
            # compatible with different OS.
            line = line.strip().replace('/', os.sep).split(',')
            data_info = dict(
                video_path=line[0],
                ann_path=line[1],
                start_frame_id=int(line[2]),
                end_frame_id=int(line[3]),
                framename_template='%04d.jpg')
            data_infos.append(data_info)
        print(f'Video-SAR dataset loaded! ({time.time() - start_time:.2f} s)')

        if split in ('left', 'right'):
            data_infos = [data for data in data_infos
                          if split in data['video_path']]
        elif split == 'train':
            data_infos = [data for data in data_infos
                          if data['video_path'].split('/')[-1]
                          in self.selected_train_targets]
        elif split == 'test':
            data_infos = [data for data in data_infos
                          if data['video_path'].split('/')[-1]
                          not in self.selected_train_targets]

        return data_infos

    def evaluate(self, results, metric=None, iou_th=None, logger=None):
        """Default evaluation standard is OPE.

        Args:
            results (dict(list[ndarray])): tracking results. The ndarray is in
                (x1, y1, x2, y2, score) format.
            metric (list, optional): defaults to ['track'].
            iou_th (ndarray | list | tuple | None): The `iou_th` of
                `success_overlap`. If `None`, the `iou_th` will be
                `np.arange(0, 1.05, 0.05)`. if list or tuple is input,
                the `iou_th` will be `np.arange(*iou_th)`. Default to `None`.
            logger (logging.Logger | str | None, optional): defaults to None.
        """

        if metric is None:
            metric = ['track']

        if isinstance(iou_th, (list, tuple)):
            iou_th = np.arange(*iou_th)

        if isinstance(metric, list):
            metrics = metric
        elif isinstance(metric, str):
            metrics = [metric]
        else:
            raise TypeError('metric must be a list or a str.')
        allowed_metrics = ['track']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported.')

        # get all test annotations
        gt_bboxes = []
        visible_infos = []
        for video_ind in range(len(self.data_infos)):
            video_anns = self.get_ann_infos_from_video(video_ind)
            gt_bboxes.append(video_anns['bboxes'])
            visible_infos.append(video_anns['visible'])

        # tracking_bboxes converting code
        eval_results = dict()
        if 'track' in metrics:
            assert len(self) == len(
                results['track_bboxes']
            ), f"{len(self)} == {len(results['track_bboxes'])}"
            print_log('Evaluate OPE Benchmark...', logger=logger)
            track_bboxes = []
            start_ind = end_ind = 0
            for num in self.num_frames_per_video:
                end_ind += num
                track_bboxes.append(
                    list(
                        map(lambda x: x[:-1],
                            results['track_bboxes'][start_ind:end_ind])))
                start_ind += num

            if not self.only_eval_visible:
                visible_infos = None
            # evaluation
            track_eval_results = eval_sot_ope(
                results=track_bboxes,
                annotations=gt_bboxes,
                visible_infos=visible_infos,
                iou_th=iou_th)
            eval_results.update(track_eval_results)

            for k, v in eval_results.items():
                if isinstance(v, float):
                    eval_results[k] = float(f'{v :.3f}')
            print_log(eval_results, logger=logger)
        return eval_results