# Modified from siamese_rpn_r50_20e_uav123.py
_base_ = './siamese_rpn/siamese_rpn_r50_20e_lasot.py'

# urls
uav123_ckpt_url = r'https://download.openmmlab.com/mmtracking/sot/' \
                  r'siamese_rpn/siamese_rpn_r50_1x_uav123/siamese_rpn_r50_20e_uav123_20220420_181845-dc2d4831.pth'  # noqa

# model settings
model = dict(
    type='SiameseRPN',
    backbone=dict(init_cfg=dict(_delete_=True)),
    test_cfg=dict(rpn=dict(penalty_k=0.1, window_influence=0.1, lr=0.5)),
    init_cfg=dict(type='Pretrained', checkpoint=uav123_ckpt_url))

data_root = 'data/'
# dataset settings
crop_size = 511
exemplar_size = 127
search_size = 255
train_pipeline = [
    dict(
        type='PairSampling',
        frame_range=100,
        pos_prob=0.8,
        filter_template_img=False),
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_label=False),
    dict(
        type='SeqCropLikeSiamFC',
        context_amount=0.5,
        exemplar_size=exemplar_size,
        crop_size=crop_size),
    dict(
        type='SeqShiftScaleAug',
        target_size=[exemplar_size, search_size],
        shift=[4, 64],
        scale=[0.05, 0.18]),
    dict(type='SeqColorAug', prob=[1.0, 1.0]),
    dict(type='SeqBlurAug', prob=[0.0, 0.2]),
    dict(type='VideoCollect', keys=['img', 'gt_bboxes', 'is_positive_pairs']),
    dict(type='ConcatSameTypeFrames'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='search')
]
data = dict(
    samples_per_gpu=28,
    workers_per_gpu=4,
    persistent_workers=True,
    samples_per_epoch=600000,
    train=dict(
        _delete_=True,
        type='VideoSARDataset',
        ann_file=data_root + 'snl_dataset/annotations/snl_infos.txt',
        img_prefix=data_root + 'snl_dataset',
        only_eval_visible=False,
        pipeline=train_pipeline,
        split='train',
        test_mode=False),
    val=dict(
        type='VideoSARDataset',
        ann_file=data_root + 'snl_dataset/annotations/snl_infos.txt',
        img_prefix=data_root + 'snl_dataset',
        only_eval_visible=False,
        split='test',
        test_mode=True),
    test=dict(
        type='VideoSARDataset',
        ann_file=data_root + 'snl_dataset/annotations/snl_infos.txt',
        img_prefix=data_root + 'snl_dataset',
        only_eval_visible=False,
        split='test',
        test_mode=True))

# learning policy
lr_config = dict(
    policy='SiameseRPN',
    lr_configs=[
        dict(type='step', start_lr_factor=0.2, end_lr_factor=1.0, end_epoch=1),
        dict(type='log', start_lr_factor=1.0, end_lr_factor=0.1, end_epoch=4),
    ])
total_epochs = 4

optimizer = dict(
    type='SGD',
    lr=0.001)

evaluation = dict(
    metric=['track'],
    interval=1,
    start=0,
    rule='greater',
    save_best='success',
    iou_th=[0, 0.50, 0.05])  # NOTICE: You can delete this line to align with CV community.  # noqa

custom_imports = dict(
    imports='sar_track',
    allow_failed_imports=False)
