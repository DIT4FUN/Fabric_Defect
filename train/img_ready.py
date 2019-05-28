from train.imgTool import readIMGInDir as readIMGInDir
from train.imgTool import imgdetection as imgdetection
from train.osTools import mkdirL as mkdirL
import train.labelTool as labelTool
from PIL import Image
import cv2 as cv


def classifyIMG(imgpath, labelpath, savepath, IMG_H, IMG_W):
    """
    预处理带瑕疵图片写入硬盘
    :param imgpath: 图片文件夹路径
    :param labelpath: 标签文件夹路径
    :return:
    """
    mkdirL(savepath, range(1, 69))
    labelL = labelTool.readclassify(labelpath)
    imgFilePathL = readIMGInDir(imgpath)
    for id, imgFilePath in enumerate(imgFilePathL):
        try:
            img_obj = imgdetection(imgFilePath)
            im = img_obj.three2one()
            im = Image.fromarray(cv.cvtColor(im, cv.COLOR_BGR2RGB))
        except:
            continue
        im = im.resize((IMG_H, IMG_W), Image.ANTIALIAS)
        imFileName = imgFilePath[imgFilePath.rindex("/") + 1:imgFilePath.rindex("/") + 7]
        try:
            label = labelL[str(imFileName)]  # 得到Label
        except:
            continue
        im.save(savepath + "/" + str(label) + "/" + str(imFileName) + ".jpg")
        if id % 10 == 0:
            print(id)

def classifyIMG(imgpath, savepath, IMG_H, IMG_W):

IMG_H = 323
IMG_W = 180
'''
classifyIMG("F:/Fabric_Defect2/train/trainData/Classified2/img",
            "F:/Fabric_Defect2/train/trainData/Classified2/label",
            'F:/Fabric_Defect2/train/trainData/Classified2/classify', IMG_H, IMG_W)
'''