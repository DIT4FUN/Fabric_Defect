'''
ImgTool
处理图片工具集合
'''
import os
from PIL import Image


def readIMGInDir(path, type=None):
    '''
    读取文件夹下所有文件的文件名和路径
    :param path: 路径
    type:指定文件类型，如果没有指定则视为jpg类型
    :return: 文件夹内所有路径+文件名
    '''
    if type is None:
        type = '.jpg'
    nameL = []  # 保存文件名
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] == type:
                nameL.append(str(os.path.join(root, file)).replace("\\", "/"))

    return nameL
    # 其中os.path.splitext()函数将路径拆分为文件名+扩展名


def imgCentreCut(filePath, savePath='./trainData/centre',block_size = 256):
    '''
    图像中心裁剪，适用于布匹类型判断
    :param filePath: 图片路径
    :return:
    '''
    fileName = filePath[filePath.rindex("/"):]
    fileName=fileName[:fileName.rindex(".jpg")]+".bmp"
    img = Image.open(filePath)
    (H, W) = img.size
    cen_img = img.crop(((H - block_size) / 2, (W - block_size) / 2, (H + block_size) / 2, (W + block_size) / 2))
    cen_img.save(savePath + fileName,'bmp')
