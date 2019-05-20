'''
ImgTool
处理图片工具集合
'''
import os
from PIL import Image
import cv2 as cv
import numpy


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


def cannyPIL(img_cv_Obj):
    '''

    :param img_PIL_Obj: pillow对象的图片
    :return: 直接返回PIL对象数据
    '''
    a = numpy.array(img_cv_Obj)
    img = cv.cvtColor(a, cv.COLOR_BGR2GRAY)
    cannyimg = cv.Canny(img, 0, 255)
    image = Image.fromarray(cv.cvtColor(cannyimg, cv.COLOR_BGR2RGB))
    return image


def imgCentreCut(filePath, savePath='./trainData/centre/', block_size=256, detection=False):
    '''
    图像中心裁剪，适用于布匹类型判断
    :param filePath: 图片路径
    savePath:保存路径 如果为None则不保存直接返回cv对象数据
    block_size:切割块大小
    detection:是否进行边缘检测
    :return:cv对象数据
    '''
    TRANSLATION = 256
    fileName = filePath[filePath.rindex("/") + 1:]
    fileName = fileName[:fileName.rindex(".jpg")] + ".png"
    try:
        img = cv.imread(filePath)
        H, W, C = img.shape
    except:
        return 0
    # img=cv.resize(img, (H*2, W * 2))
    cen_img = img[(H - block_size) // 2:(H + block_size) // 2 + TRANSLATION,
              (W - block_size) // 2 + TRANSLATION:(W + block_size) // 2 + TRANSLATION]
    # PIL方法
    # img = Image.open(filePath)
    # (H, W) = img.size
    # cen_img = img.crop(((H - block_size) / 2, (W - block_size) / 2, (H + block_size) / 2, (W + block_size) / 2))
    if detection is True:
        # cen_img=cannyPIL(cen_img)
        cen_img = cv.Scharr(cen_img, -1, 1, 0)
    if savePath is None:
        return cen_img
    else:
        # cen_img.save(savePath + fileName, 'png')
        cv.imwrite(savePath + fileName, cen_img, [int(cv.IMWRITE_PNG_COMPRESSION), 9])
