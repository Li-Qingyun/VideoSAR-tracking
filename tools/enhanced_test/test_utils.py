# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import shutil
import tempfile
import time
from collections import defaultdict

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from mmdet.core import encode_mask_results


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    fps=3,
                    show_score_thr=0.3):
    """Test model with single gpu.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool, optional): If True, visualize the prediction results.
            Defaults to False.
        out_dir (str, optional): Path of directory to save the
            visualization results. Defaults to None.
        fps (int, optional): FPS of the output video.
            Defaults to 3.
        show_score_thr (float, optional): The score threshold of visualization
            (Only used in VID for now). Defaults to 0.3.

    Returns:
        dict[str, list]: The prediction results.
    """
    model.eval()
    results = defaultdict(list)
    dataset = data_loader.dataset
    prev_img_meta = None
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        batch_size = data['img'][0].size(0)
        if show or out_dir:
            assert batch_size == 1, 'Only support batch_size=1 when testing.'
            img_tensor = data['img'][0]
            img_meta = data['img_metas'][0].data[0][0]
            img = tensor2imgs(img_tensor, **img_meta['img_norm_cfg'])[0]

            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]

            ori_h, ori_w = img_meta['ori_shape'][:-1]
            img_show = mmcv.imresize(img_show, (ori_w, ori_h))

            if out_dir:
                out_file = osp.join(out_dir, img_meta['ori_filename'])
            else:
                out_file = None

            model.module.show_result(
                img_show,
                result,
                gt=data['gt_bboxes'],
                show=show,
                out_file=out_file,
                score_thr=show_score_thr)

            # Whether need to generate a video from images.
            # The frame_id == 0 means the model starts processing
            # a new video, therefore we can write the previous video.
            # There are two corner cases.
            # Case 1: prev_img_meta == None means there is no previous video.
            # Case 2: i == len(dataset) means processing the last video
            need_write_video = (
                prev_img_meta is not None and img_meta['frame_id'] == 0
                or i == len(dataset))
            if out_dir and need_write_video:
                prev_img_prefix, prev_img_name = prev_img_meta[
                    'ori_filename'].rsplit(os.sep, 1)
                prev_img_idx, prev_img_type = prev_img_name.split('.')
                prev_filename_tmpl = '{:0' + str(
                    len(prev_img_idx)) + 'd}.' + prev_img_type
                prev_img_dirs = f'{out_dir}/{prev_img_prefix}'
                prev_img_names = sorted(os.listdir(prev_img_dirs))
                prev_start_frame_id = int(prev_img_names[0].split('.')[0])
                prev_end_frame_id = int(prev_img_names[-1].split('.')[0])

                mmcv.frames2video(
                    prev_img_dirs,
                    f'{prev_img_dirs}/out_video.mp4',
                    fps=fps,
                    fourcc='mp4v',
                    filename_tmpl=prev_filename_tmpl,
                    start=prev_start_frame_id,
                    end=prev_end_frame_id,
                    show_progress=False)

            prev_img_meta = img_meta

        for key in result:
            if 'mask' in key:
                result[key] = encode_mask_results(result[key])

        for k, v in result.items():
            results[k].append(v)

        for _ in range(batch_size):
            prog_bar.update()

    return results