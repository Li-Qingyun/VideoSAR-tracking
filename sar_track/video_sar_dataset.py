import os
import time
from mmtrack.datasets import DATASETS, BaseSOTDataset


@DATASETS.register_module()
class VideoSARDataset(BaseSOTDataset):
    """Video SAR interface of single object tracking.
    Modified from `UAV123Dataset
    <>`_.
    """

    def load_data_infos(self, split='all'):
        """Load dataset information.

        Args:
            split (str, optional): Dataset split, can be 'left', 'right', and
                'all'. Defaults to 'all'.

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
        assert split in ('all', 'left', 'right')

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

        if split != 'all':
            data_infos = [data for data in data_infos
                          if split in data['video_path']]

        return data_infos