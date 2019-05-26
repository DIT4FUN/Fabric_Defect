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


def mkdirL(path, namelist, de=False):
    '''
    批量创建目录
    :param path: 主目录
    :param namelist: 子目录命名列表
    :param de: 是否删除旧文件
    :return: None
    '''
    for i in namelist:
        mkdir(path + "/" + str(i), de=de)


