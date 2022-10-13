# Video-SAR Object Tracking

Implement object tracking task of SAR Video. This repo is built on [mmtracking](https://github.com/open-mmlab/mmtracking) (OpenMMLab 1.0 series). 

Author: Qingyun Li (21B905003@stu.hit.edu.cn)    Mentor: Yushi Chen (chenyushi@hit.edu.cn)



The supported Video-SAR data:

- SNL.mp4

The evaluated algorithm:

- [SiameseRPN++](https://github.com/open-mmlab/mmtracking/tree/master/configs/sot/siamese_rpn) 



## Prepare Environment

- [mim](https://github.com/open-mmlab/mim)
- [mmtracking](https://github.com/open-mmlab/mmtracking)

```bash
# create and activate env
conda create -n sar-tracking python=3.7
conda activate sar-tracking

# install pytorch and torchvision
conda install -c pytorch pytorch torchvision

# install mmtracking
pip install openmim
mim install mmtracking
```



## Prepare Data

Chinese users can refer to [FlowUS Blog](https://flowus.cn/71022a70-d2a1-4145-bb02-1046624fead3) to know about our data annotation. We used [LabelBee](https://github.com/open-mmlab/labelbee) of OpenMMLab. 

You can run `python get_frames.py` and annotated by yourself, or contact Menglu Zhang or Yushi Chen for annotated data (I do not have the authority).

When your data folder `snl_dataset` has been prepared, you can run `python convert_videosar_ann.py` to convert the LabelBee annotation to mmtracking annotation.



## Training and Evaluating

We use mmf to conduct training.

```bash
# training
python train.py

# sot demo
python demo_sot.py configs/siamese_rpn_r50_20e_videosar.py --input data/SNL_Trim.mp4 --checkpoint work_dirs/xxx/ckpt.pth --show --fps 30  # sot on Video File
python demo_sot.py configs/siamese_rpn_r50_20e_videosar.py --input data/snl_dataset/data_seq/right_target01 --checkpoint work_dirs/xxx/epoch_3.pth --show --fps 30  # sot on Frame Folder
```

