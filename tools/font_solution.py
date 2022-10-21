"""
————————————————
版权声明：本文为CSDN博主「机器想学习」的原创文章，遵循CC4.0
BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/kuwola/article/details/122409200
"""

from matplotlib.font_manager import FontManager

mpl_fonts = set(f.name for f in FontManager().ttflist)

print('all font list get from matplotlib.font_manager:')
for f in sorted(mpl_fonts):
    print('\t' + f)
