import sys
import time


def progress_print(now_index, sum_index):
    """
    进度条
    """
    time.sleep(0.1)
    percentage = now_index / sum_index
    # 输出进度条
    style = "+"
    stdout_obj=sys.stdout
    stdout_obj.write('\rPercentage of progress:{:.2%}'.format(percentage))
    if now_index == sum_index:
        print("\n----------ImgPretreatment Done!-----------")

progress_bar_obj = sys.stdout
for i in range(100):
    progress_print( i, 150)

