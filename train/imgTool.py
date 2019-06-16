'''
ImgTool
处理图片工具集合
'''
import os
from PIL import Image, ImageFont, ImageDraw
import cv2 as cv
import numpy
import traceback

fontP = "./font/1.ttf"


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

def saveIMGFilePathL(dirPath, savePath, oriIMGType="jpg", for_test=False):
    '''
    创建图像列表-需要为LabelME创建的文件夹格式
    :param dirPath: LabelME生成文件夹格式
    :param savePath:图像列表的不以/为结尾保存位置
    :param for_test=False：是否用于测试 测试则不生成对应蒙版图片列表
    :return:
    '''
    if for_test is not False:
        with open(savePath + "/test.list", "w") as f:
            pathL = readIMGInDir(dirPath + "/JPEGImages", onle_name=True)
            for file in pathL:
                f.writelines("JPEGImages/" + file + "\n")
            print("写入量", len(pathL))
    else:
        # oriPathL = readIMGInDir(dirPath + "/JPEGImages", onle_name=True)
        maskPathL = readIMGInDir(dirPath + "/SegmentationClassPNG", type="png", onle_name=True)
        with open(savePath + "/train.list", "w") as f:
            count = 0
            for mask in maskPathL:
                f.writelines("JPEGImages/" + mask[:-3] + oriIMGType + " " + "SegmentationClassPNG/" + mask + "\n")
                count += 1
            print("写入量", count)


# saveIMGFilePathL("./trainData/20181024/out", "./trainData/20181024/out", for_test=True)


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
            im = a.three2one()
            cv.imshow("1", im)
            cv.waitKey()
        except:
            print(traceback.format_exc())
    print("Done")


# debugIMG("./trainData/ori2/")

def cut_box_for_infer(img, quick=False):
    '''
    GT-CutR模型 预测图片预处理工具
    :param img: PIL_Obj
    :param quick = False:极速模式 关闭
    :return: imgL 剪裁后图片列表
    '''
    # 参数表
    boxsixe = [240, 240]  # 2倍的测试框
    input_img = (1200, 2448)  # 输入图片
    check_input = (2448, 1200)
    assert (check_input == img.size), "输入图片不符合(1200, 2448)尺寸！"
    vbox_input_img = [960, 2160]  # 虚拟图片大小
    ext = 0.  # ext: 扩充偏移量[0-0.5]
    extup = 0.  # extup:上下偏移量[0.3-1]

    extupL = 144 * (1 - extup)
    img = img.crop((extupL, 120, vbox_input_img[1] + extupL, vbox_input_img[0] + 120))  # W H
    p = 1
    input_img = img
    # 步长
    long = boxsixe[0] // 2
    if quick is True:
        p = 2
        ext *= 0.5

    '''
    选择框处理工具
    :param input_img: PIL对象
    :return: 小块图像列表
    '''
    mini_imgL = []
    box_W = (2 * vbox_input_img[1]) // boxsixe[0]
    box_H = (2 * vbox_input_img[0]) // boxsixe[0]
    # print(box_W, box_H) 18 8
    for list_W in range(box_W):
        for list_H in range(0, box_H, p):
            if list_W % 2 == 1:
                if list_W == box_W - 1:
                    continue
                box = (list_W * long + ext * boxsixe[0], list_H * long, (list_W + 1) * long + ext * boxsixe[0],
                       (list_H + 1) * long)

            else:
                box = (list_W * long, list_H * long, (list_W + 1) * long, (list_H + 1) * long)
            mini_img = input_img.crop(box)
            mini_imgL.append(mini_img)
    return mini_imgL


'''
imgF="./testData/075353_1_0.jpg"
img=Image.open(imgF)
cut_box_for_infer(img)
'''


def drawIMG(dirP, imgname):
    '''

    :param imgP:  图片文件路径
    :return: PIL对象
    '''
    imgP = dirP + "/" + imgname
    labelP = dirP + "/info/" + imgname + ".txt"
    imgP = Image.open(imgP)
    vbox_input_img = [960, 2160]  # 虚拟图片大小
    imgP = imgP.crop((0, 120, vbox_input_img[1], vbox_input_img[0] + 120))
    draw = ImageDraw.Draw(imgP)
    with open(labelP, "r") as f:
        info = f.read().split("\n")[:-1]
        for i in info:
            L = i.split("-")
            W = int(L[0]) - 1
            H = int(L[1]) - 1
            strL = L[2] + "/" + L[3]
            font = ImageFont.truetype(fontP, 15, encoding="utf-8")
            draw.text((60 + 120 * W, 120 + 240 * H), strL, fill=200, font=font)
    return imgP


a = drawIMG("./testData", "222443_1_5Y.jpg")
a.show()
