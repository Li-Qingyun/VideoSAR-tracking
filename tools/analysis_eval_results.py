import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import pandas as pd
from pylab import mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False


EVAL_META_PATH = r'work_dirs/siamese_rpn_r50_20e_videosar/eval_out/eval_meta_2022-10-21-09_12_16.pth'  # noqa
SAVE_DIR = r'work_dirs/siamese_rpn_r50_20e_videosar/eval_out'


def draw_plog_fig(threshold, value, x_name, y_name, save_path):
    sns.set(style='darkgrid')
    df = pd.DataFrame({x_name: threshold, y_name: value})
    g = sns.relplot(x=x_name, y=y_name, kind='line', data=df)
    g.fig.autofmt_xdate()
    plt.savefig(save_path)


if __name__ == '__main__':
    eval_meta = torch.load(EVAL_META_PATH)

    success = eval_meta['success']
    precision = eval_meta['precision']
    norm_precision = eval_meta['norm_precision']

    success_results = np.mean(np.array(success['results']), axis=0)
    precision_results = np.mean(np.array(precision['results']), axis=0)
    norm_precision_results = np.mean(np.array(norm_precision['results']), axis=0)

    draw_plog_fig(success['threshold'], success_results,
                  success['threshold_name'], 'success',
                  # '阈值', '成功率',
                  os.path.join(SAVE_DIR, 'success.png'))
    draw_plog_fig(precision['threshold'], precision_results,
                  precision['threshold_name'], 'precision',
                  # '阈值', '精确度',
                  os.path.join(SAVE_DIR, 'precision.png'))
    draw_plog_fig(norm_precision['threshold'], norm_precision_results,
                  norm_precision['threshold_name'], 'norm_precision',
                  # '阈值', '归一化精确度',
                  os.path.join(SAVE_DIR, 'norm_precision.png'))