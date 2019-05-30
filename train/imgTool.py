'''
ImgTool
处理图片工具集合
'''
import os
from PIL import Image
import cv2 as cv
import numpy
import traceback


def readIMGInDir(path, type=None, onle_name=False):
    '''
    读取文件夹下所有文件的文件名和路径
    :param path: 路径
    type:指定文件类型，如果没有指定则视为jpg类型
    :return: nameL:文件夹内所有路径+文件名 './trainData/ori1/20181024/000030_1_0.jpg' or '000030_1_0.jpg'
    '''
    if type is None:
        type = '.jpg'
    else:
        type = "." + type
    nameL = []  # 保存文件名
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] == type:
                if onle_name is True:
                    nameL.append(str(file).replace("\\", "/"))
                else:
                    nameL.append(str(os.path.join(root, file)).replace("\\", "/"))
    return nameL
    # 其中os.path.splitext()函数将路径拆分为文件名+扩展名


# print(readIMGInDir("./trainData/"))

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


def imgCentreCut(filePath, savePath='./trainData/centre', block_size=256, detection=False):
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
        W, H, C = img.shape
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
    cen_img = cen_img[:32, :32]

    if savePath is None:
        return cen_img
    else:
        # cen_img.save(savePath + fileName, 'png')
        cv.imwrite(savePath + "/" + fileName, cen_img, [int(cv.IMWRITE_PNG_COMPRESSION), 9])


class imgdetection:
    '''
    瑕疵检测预处理
    仅处理单个图片
    '''
    # 采样倍率
    opt = [0.01, 0.065, 0.3, 0.8, 1, 1.4, 2]

    def __init__(self, imgFilePath, opt=None):
        self.imgFilePath = imgFilePath
        self.img = cv.imread(imgFilePath, 0)
        if opt is not None:
            self.opt = opt

    def _imgresize(self, size_num):
        '''
        图像大小调整
        :param size_num: 缩放倍数
        :return: cv对象
        '''
        H, W = self.img.shape
        img = cv.resize(self.img, (int(W * size_num), int(H * size_num)))
        return img

    def _imgcanny(self, img):
        '''
        Canny边缘检测
        :param img: cv对象
        :return: cv对象
        '''
        # blur = cv.GaussianBlur(img, (3, 3), 0)  # 高斯滤波降噪   参数  内核 偏差
        edge = cv.Canny(img, 30, 65)  # 30最小阈值 70最大阈值
        return edge

    def detection(self):
        '''
        采样批处理工具
        :return: 图像字典 {[int]采样倍数:cv对象}
        '''
        imgL = []
        for i in self.opt:
            img = self._imgresize(i)  # 调整大小
            img = self._imgcanny(img)
            imgL.append((i, img))
        imgL = dict(imgL)
        return imgL

    def edgeFind(self):
        '''
        :return: [4339, 4483, 0, 2400] 起始W位置，结束W位置，起始H，结束H
        '''

        img_cv_obj = self.detection()[self.opt[-1]]
        H, W = img_cv_obj.shape
        im = cv.resize(img_cv_obj, (1020, 500))
        finalbox = [0, 0, 0, 0]

        sumA = 0  # 计数器 超过指定值即为轮廓
        for i in range(1000):
            s = 0
            for ii in range(150, 300):
                if im[ii, i] != 0 and s <= 10:
                    s += 1
                    for iii in range(350, 400):
                        if im[iii, i] != 0 and sumA <= 50:
                            sumA += 1
                        if sumA == 50:
                            finalbox = [i - 30, i + 30, 0, 500]
                            break
        finalbox = [int(i * (W / 1020)) for i in finalbox]
        return finalbox

    def roidel(self):
        '''
        去除布匹边缘
        :return: 图像字典 {[int]采样倍数:cv对象}
        '''
        box = self.edgeFind()
        imgL = []
        for op in self.opt:
            img_cv_obj = self.detection()[op]  # 取图片
            # roi=[int(box[2]//2*op),int(box[3]//2*op), int(box[0]//2*op),int(box[1]//2*op)]
            img_cv_obj[int(box[2] / 2 * op):int(box[3] / 2 * op), int(box[0] / 2 * op):int(box[1] / 2 * op)] = 0
            imgL.append((op, img_cv_obj))
        imgL = dict(imgL)
        return imgL

    def three2one(self):
        '''
        通道信息3合1
        :return: cv对象
        '''
        img_1 = self.detection()[self.opt[1]]
        img_2 = self.detection()[self.opt[2]]
        img_3 = self.detection()[self.opt[3]]
        H, W = img_3.shape
        img_1 = cv.resize(img_1, (W, H))
        img_2 = cv.resize(img_2, (W, H))
        im = [img_1, img_2, img_3]
        img = cv.merge(im)
        return img


def debugIMG(path):
    '''
    Debug-目录下的图片提取为指定类型
    :param path: 图片目录
    :return: None
    '''
    from train.osTools import mkdirL
    # path="./trainData/ori/img/"
    mkdirL(path, imgdetection.opt, de=True)
    img_Name = readIMGInDir(path)
    for i in img_Name:
        '''
        try:
            a = imgdetection(i)
            imgL = a.detection()
            for ii in range(len(imgdetection.opt)):
                p = path + str(imgdetection.opt[ii]) + "/" + str(i[i.rindex("/"):]) + '.jpg'
                cv.imwrite(p, imgL[imgdetection.opt[ii]])
            print(i, "--OK!")
        except:
            
        '''
        try:
            a = imgdetection(i)
            im=a.three2one()
            cv.imshow("1",im)
            cv.waitKey()
        except:
            print(traceback.format_exc())
    print("Done")

#debugIMG("./trainData/ori2/")
