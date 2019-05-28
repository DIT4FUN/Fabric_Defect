import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt
from train.torNN import TorNN
from train.imgTool import readIMGInDir as readIMGInDir
from train.imgTool import imgdetection as imgdetection
from train.osTools import mkdirL as mkdirL
from train.osTools import readDirName as readDirName
from pylab import mpl

# 参数表
place = fluid.CPUPlace()
IMG_H = 323  # 输入网络图像大小1958 960
IMG_W = 180
TRAINNUM = 50  # 训练次数
READIMG = 155

# 指定路径
# 路径除root外均不带"/"后缀
path = './'
testModelPath = path + "model/defectBase.model22INF0.821192"
imgPath = path + "trainData/Classified"
dirL = readDirName(imgPath)  # 分类图片路径数据

# 参数初始化
exe = fluid.Executor(place)


def readIMG(imgFilePath):
    img_obj = imgdetection(imgFilePath)
    im = img_obj.three2one()
    im = Image.fromarray(cv.cvtColor(im, cv.COLOR_BGR2RGB))
    im = im.resize((IMG_H, IMG_W), Image.ANTIALIAS)
    # im.show()
    im = np.array(im).reshape(1,3, IMG_W, IMG_H).astype(np.float32)
    # im = im / 255.0 * 2.0 - 1.0
    return im


[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=testModelPath,
                                                                                      executor=exe)

img = readIMG("F:/Fabric_Defect2/train/trainData/ori1/1.jpg")
results = exe.run(inference_program,
                  feed={feed_target_names[0]: img},
                  fetch_list=fetch_targets)
lab = np.argsort(results)[0][0][-1]
print(results)
print(lab)
