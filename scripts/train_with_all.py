from mim import run

run(package='mmtrack', command='train',
    other_args=(r'configs/siamese_rpn_r50_20e_videosar_all_train_all_val.py',
                # '--work-dir', 'tmp'
                ))