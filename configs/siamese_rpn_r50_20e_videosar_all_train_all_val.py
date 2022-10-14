_base_ = './siamese_rpn_r50_20e_videosar.py'

data = dict(
    train=dict(split='all'),
    val=dict(split='all'),
    test=dict(split='all')
)