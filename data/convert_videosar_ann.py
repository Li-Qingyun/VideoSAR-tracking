from pathlib import Path
from check_coco_ann import check_coco_ann
from convert_utils import Pathtype, coco2tracking


COCO_ANN_FOLDER = r'./labelbee_outputs'
DATA_SEQ_FOLDER = r'./snl_dataset/data_seq'
OUT_ANN_FOLDER = r'./snl_dataset/anno'
OUT_INFO_PATH = r'./snl_dataset/annotations/snl_infos.txt'


def convert_all_ann(coco_ann_folder: Pathtype,
                    data_seq_folder: Pathtype,
                    out_ann_folder: Pathtype,
                    out_info_path: Pathtype) -> None:
    if isinstance(coco_ann_folder, str):
        coco_ann_folder = Path(coco_ann_folder)
    if isinstance(data_seq_folder, str):
        data_seq_folder = Path(data_seq_folder)
    if isinstance(out_ann_folder, str):
        out_ann_folder = Path(out_ann_folder)
    if isinstance(out_info_path, str):
        out_info_path = Path(out_info_path)
    assert coco_ann_folder.exists(), coco_ann_folder
    assert data_seq_folder.exists(), data_seq_folder
    out_ann_folder.mkdir(exist_ok=True)
    out_info_path.parent.mkdir(exist_ok=True, parents=True)

    check_coco_ann(coco_ann_folder, data_seq_folder)
    target_frame_folders = sorted(list(data_seq_folder.iterdir()))

    tip_msg = 'The format of each line in this txt is ' \
              '(video_path,annotation_path,start_frame_id,end_frame_id)'
    write_info(tip_msg, out_info_path)
    for target_frame_folder in target_frame_folders:
        target_name = target_frame_folder.name
        frames = sorted(list(target_frame_folder.iterdir()))

        coco_ann_path = coco_ann_folder / f'{target_name}-coco.json'
        out_txt_path = out_ann_folder / f'{target_name}.txt'

        coco2tracking(coco_ann_path, out_txt_path)

        # NOTE All the path is without prefix 'snl_dataset'
        info = ['/'.join(target_frame_folder.parts[1:]),  # frame folder
                '/'.join(out_txt_path.parts[1:]),  # ann txt
                str(int(frames[0].stem)),  # start frame id
                str(int(frames[-1].stem))]  # end frame id
        write_info(info, out_info_path)

    print('CONVERT DONE!')


def write_info(info: 'list | str', info_path: Pathtype):
    if isinstance(info_path, str):
        info_path = Path(info_path)
    msg = ','.join(info) if isinstance(info, list) else info
    print(msg)
    with info_path.open('a') as f:
        f.write(msg + '\n')


if __name__ == '__main__':
    convert_all_ann(COCO_ANN_FOLDER, DATA_SEQ_FOLDER,
                    OUT_ANN_FOLDER, OUT_INFO_PATH)