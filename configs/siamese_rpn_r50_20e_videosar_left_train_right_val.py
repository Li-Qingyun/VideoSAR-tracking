_base_ = './siamese_rpn_r50_20e_videosar.py'

data = dict(
    train=dict(split='left'),
    val=dict(split='right'),
    test=dict(split='right')
)