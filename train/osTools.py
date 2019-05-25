import os
import shutil


def mkdir(path, de=False):
    '''
    判断是否路径存在
    :param path: 文件路径
    :return: None
    '''
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        if de is True:
            shutil.rmtree(path)
            os.makedirs(path)
