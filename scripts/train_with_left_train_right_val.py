from mim import run

run(package='mmtrack', command='train',
    other_args=(r'configs/siamese_rpn_r50_20e_videosar_left_train_right_val.py',
                # '--work-dir', 'tmp'
                ))