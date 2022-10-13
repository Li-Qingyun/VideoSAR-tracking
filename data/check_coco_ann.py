from pathlib import Path
from convert_utils import coco_ann_2_box_dict, Pathtype


COCO_ANN_FOLDER = r'./labelbee_outputs'
DATA_SEQ_FOLDER = r'./snl_dataset/data_seq'


def check_coco_ann(coco_ann_folder: Pathtype,
                   data_seq_folder: Pathtype) -> None:
    if isinstance(coco_ann_folder, str):
        coco_ann_folder = Path(coco_ann_folder)
    if isinstance(data_seq_folder, str):
        data_seq_folder = Path(data_seq_folder)
    assert coco_ann_folder.exists(), coco_ann_folder
    assert data_seq_folder.exists(), data_seq_folder

    CHECK_SUCCESS = 1
    for target_frame_folder in data_seq_folder.iterdir():
        target_name = target_frame_folder.name
        img_path_list = list(target_frame_folder.iterdir())
        num_frames = len(img_path_list)

        coco_ann_path = coco_ann_folder / f'{target_name}-coco.json'
        assert coco_ann_path.exists(), coco_ann_path

        box_dict = coco_ann_2_box_dict(coco_ann_path)
        num_frames_check = len(box_dict)

        if not num_frames == num_frames_check:
            print(f'{target_name}: There are {num_frames} frames in DATA_SEQ, '
                  f'however {num_frames_check} boxes in {coco_ann_path}.')
            CHECK_SUCCESS = 0

    assert CHECK_SUCCESS, 'Please keep the num_frames in DATA_SEQ ' \
                          'and num_boxes in COCO_JSON are the same.'
    print('CHECK DONE!')


if __name__ == '__main__':
    check_coco_ann(COCO_ANN_FOLDER, DATA_SEQ_FOLDER)