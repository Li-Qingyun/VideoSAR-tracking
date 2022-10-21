from mim import run

run(package='mmtrack', command='test',
    other_args=(r'configs/siamese_rpn_r50_20e_videosar.py',
                '--checkpoint', r'work_dirs/siamese_rpn_r50_20e_videosar/best_success_epoch_2.pth',  # noqa
                # '--work-dir', 'tmp'
                '--eval', 'track',
                # '--show-dir', r'work_dirs/siamese_rpn_r50_20e_videosar/inference_out/',  # noqa
                '--cfg-options',
                r'evaluation.meta_save_dir=work_dirs/siamese_rpn_r50_20e_videosar/eval_out/',  # noqa
                ))