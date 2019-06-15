import numpy as np
from PIL import Image
import imgTool

IMG_SHAPE = [1, 120, 120]


def dataReaderA(path):
    '''
    数据集读取函数
    :param path: 图片路径
    :return: reader对象
    '''

    def reader():
        imgL = imgTool.readIMGInDir(path)
        img_nameL = imgTool.readIMGInDir(path)
        for img, name in zip(imgL, img_nameL):
            im = Image.open(img).convert('L')
            im = np.array(im).reshape(IMG_SHAPE).astype(np.float32)
            label = name.split("-")[-1][:-4]
            yield im, int(label)

    return reader
