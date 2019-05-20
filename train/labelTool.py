import os

def readIMGInDir(path, type=None):
    '''
    读取文件夹下所有文件的文件名和路径
    :param path: 路径
    type:指定文件类型，如果没有指定则视为jpg类型
    :return: 文件夹内所有路径+文件名
    '''
    if type is None:
        type = '.txt'
    nameL = []  # 保存文件名
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] == type:
                nameL.append(str(os.path.join(root, file)).replace("\\", "/"))
    return nameL