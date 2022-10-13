from mim import run

run(package='mmtrack', command='test',
    other_args=(r'configs/siamese_rpn_r50_20e_videosar.py',
                '--checkpoint', r'work_dirs/siamese_rpn_r50_20e_videosar_from_scratch.py/latest.pth',
                # '--work-dir', 'tmp'
                r'--eval', 'track'
                ))