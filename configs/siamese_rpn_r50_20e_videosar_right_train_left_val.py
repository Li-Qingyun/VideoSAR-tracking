_base_ = './siamese_rpn_r50_20e_videosar.py'

data = dict(
    train=dict(split='right'),
    val=dict(split='left'),
    test=dict(split='left')
)