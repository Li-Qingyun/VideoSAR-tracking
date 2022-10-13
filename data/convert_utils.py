import json
from typing import Union
from pathlib import Path, PosixPath
Pathtype = Union[str, Path, PosixPath]


def coco2tracking(coco_ann_path: Pathtype, out_txt_path: Pathtype) -> list:
    box_dict = coco_ann_2_box_dict(coco_ann_path)
    box_list = box_dict_2_box_list(box_dict)
    write_box_on_txt(box_list, out_txt_path)
    return box_list


def coco_ann_2_box_dict(coco_ann_path: Pathtype) -> dict:
    coco_ann = json.load(open(coco_ann_path, 'r'))['annotations']
    box_dict = {ann['image_id']: ann['bbox'] for ann in coco_ann}
    assert len(coco_ann) == len(box_dict)
    return box_dict


def box_dict_2_box_list(box_dict: dict) -> list:
    index_list = sorted(list(box_dict.keys()))
    num_frames = len(box_dict)
    assert max(index_list) - min(index_list) == num_frames - 1
    out_box_list = [box_dict[i] for i in index_list]
    return out_box_list


def write_box_on_txt(box_list: list, txt_path: Pathtype) -> None:
    with Path(txt_path).open("a") as f:
        for box in box_list:
            f.write(f"{box[0]},{box[1]},{box[2]},{box[3]}\n")


if __name__ == '__main__':
    coco2tracking(
        'VideoSAR-tracking/labelbee_outputs/left_target17-coco.json',
        'VideoSAR-tracking/labelbee_outputs/left_target17.txt')