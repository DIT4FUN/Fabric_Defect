'''
ImgTool
处理图片工具集合
'''
import os
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import traceback

fontP = "./font/1.ttf"


def readIMGInDir(path, type=None, onle_name=False):
    """
    读取文件夹下所有文件的文件名和路径
    :param path: 路径
    type:指定文件类型，如果没有指定则视为jpg类型
    :return: nameL:文件夹内所有路径+文件名 './trainData/ori1/20181024/000030_1_0.jpg' or '000030_1_0.jpg'
    """
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
    """
    创建图像列表-需要为LabelME创建的文件夹格式
    :param dirPath: LabelME生成文件夹格式
    :param savePath:图像列表的不以/为结尾保存位置
    :param for_test=False：是否用于测试 测试则不生成对应蒙版图片列表
    :return:
    """
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

class ImgPretreatment:
    """

    图像预处理类
    传入图片文件夹路径后，对index下的图片进行预处理
    批量处理代码：
        # 创建工具对象并传入相关参数
        tool_img_pretreatment=ImgPretreatment(...)
        # 处理所有图像
        for index in range(tool_img_pretreatment.__len__())
            # 初始化当前index图像
            tool_img_pretreatment.img_init(index)
            # 对图像进行操作
            tool_img_pretreatment.img_cut_color()
            tool_img_pretreatment.img_xxx()
            tool_img_pretreatment.img_xxx()
            ...
            # 获取最终处理好的图像(Pillow对象)
            final_img_pil_obj = tool_img_pretreatment.req_final_img()

    Tips:
    1、第一次进行图像减颜色均值操作时过程较长，并非卡顿。
    2、如果要统一数据集尺寸，则最好放在所有对图像的操作之前，初始化图像操作之后。这样更有利运行速度。

    参数保存:
    在获取颜色均值后会保一个参数文件'img_pretreatment.txt'记录参数,，如果数据集总量、第一个图像
    """

    def __init__(self, all_img_path, mean_color_num=500, img_expected_size=None, img_type="jpg", read_img_type='L',
                 ignore_log=False, debug=True):
        """
        :param all_img_path: 图像文件路径
        :param mean_color_num:颜色均值采样数
        :paramimg_expected_size:期望的数据集图像统一的尺寸
        :param img_type: 图像文件类型，默认jpg格式
        :param read_img_type:图像读取通道类型 默认为灰度
        :param ignore_log:是否忽略之前参数文件
        """

        print(debug and "----------ImgPretreatment Start!----------")
        self.len_img = len(all_img_path)
        self.img_file_path = readIMGInDir(all_img_path, img_type)
        self.mean_color_num = mean_color_num
        self.img_type = img_type
        self.read_img_type = read_img_type
        self.debug = debug
        self.img_expected_size = img_expected_size

        self.shape = Image.open(self.img_file_path[0]).shape
        # Flag 变量

        self.color_mean_flag = False
        if ignore_log is False:
            try:
                with open("./img_pretreatment.txt", "r", encoding="utf-8") as f:
                    info = f.read().split("-")
                    if (info[0] == self.read_img_type and info[1] == self.len_img and info[2] == self.mean_color_num and
                            info[3] == self.img_expected_size):
                        self.color_mean_flag = True
                        print(debug and "Load Log Successfully!")
            except:
                pass
        # 当前进程变量
        self.now_index = 1
        self.now_img_obj = Image.open(self.img_file_path[0])
        print(debug and "Data Read Successfully Number:", self.len_img)

    def img_init(self, index):
        """
        图像初始化
        :param index: 需要初始化图像的索引
        :return:
        """
        if index is not self.now_index:
            try:
                self.now_img_obj = Image.open(self.img_file_path[0]).convert(self.read_img_type)
                self.now_index = index
            except:
                print(traceback.format_exc())

    def __color_mean_start(self):
        """
        颜色均值获取
        :return:颜色均值
        """

        sum_img_numpy = np.zeros((1, 1, 1), dtype=np.float)
        if self.read_img_type is "L":
            sum_img_numpy = np.zeros(1, dtype=np.float)

        self.__color_mean = [0, 0, 0]
        mean_color_num = self.mean_color_num
        success_num = 1
        for id, imgP in enumerate(self.img_file_path):
            im = Image.open(imgP).convert(self.read_img_type)
            if self.shape is not im.shape:
                mean_color_num += 1
                continue
            try:
                sum_img_numpy += np.mean(np.asarray(im), axis=0)
                success_num += 1
            except:
                print(traceback.format_exc())
            if id == mean_color_num or id == self.len_img:
                self.__color_mean = np.around((sum_img_numpy / success_num), decimals=3).tolist()
                self.color_mean_flag = True

                with open("./img_pretreatment.txt", "w", encoding="utf-8") as f:
                    f.write(
                        str(self.read_img_type) + "-" + str(self.len_img) + "-" + str(self.mean_color_num) + "-" + str(
                            self.img_expected_size))
                if self.debug is True:
                    print("Color Mean :", self.__color_mean)
                    print("Write Log --Done!")
                return self.__color_mean

    def img_cut_color(self):
        self.now_img_obj = Image.fromarray(np.asarray(self.now_img_obj) - self._req_color_mean())

    def img_only_one_shape(self):
        """
        传入图片将修正为数据集统一的尺寸
        :return:
        """
        pass
        # 暂未设计算法

    def img_resize(self):
        pass

    def req_final_img(self):
        return self.now_img_obj

    def _req_color_mean(self):
        if self.color_mean_flag:
            return self.__color_mean
        else:
            return self.__color_mean_start()

    def __len__(self):
        return self.len_img


def cut_box_for_infer(img, quick=False):
    """
    GT-CutR模型 预测图片预处理工具
    :param img: PIL_Obj
    :param quick = False:极速模式 关闭
    :return: imgL 剪裁后图片列表
    """
    # 参数表
    boxsixe = [240, 240]  # 2倍的测试框
    input_img = (1200, 2448)  # 输入图片
    check_input = (2448, 1200)
    assert (check_input == img.size), "输入图片" + str(img.size) + "不符合(2448,1200)尺寸！"
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

    """
    选择框处理工具
    :param input_img: PIL对象
    :return: 小块图像列表
    """
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


"""
imgF="./testData/075353_1_0.jpg"
img=Image.open(imgF)
cut_box_for_infer(img)
"""


def drawIMG(dirP, imgname, quickMode=True):
    """
    图片绘制工具
    :param imgP:  图片文件路径
    :param imgname:  图片文件名称
    :return: PIL对象
    """
    imgP = dirP + "/" + imgname
    labelP = dirP + "/info/" + imgname + ".txt"
    imgP = Image.open(imgP)
    vbox_input_img = [960, 2160]  # 虚拟图片大小
    imgP = imgP.crop((0, 120, vbox_input_img[1], vbox_input_img[0] + 120))
    imgP = imgP.convert('RGB')
    draw = ImageDraw.Draw(imgP)
    with open(labelP, "r") as f:
        info = f.read().split("\n")[:-1]
        for i in info:
            L = i.split("-")
            W = int(L[0]) - 1
            H = int(L[1]) - 1
            strL = L[2] + "/" + L[3]
            font = ImageFont.truetype(fontP, 15, encoding="utf-8")
            if quickMode is True:
                draw.text((60 + 120 * W, 120 + 240 * H), strL, fill="red", font=font)
            else:
                draw.text((60 + 120 * W, 120 + 120 * H), strL, fill="red", font=font)
    return imgP

# a = drawIMG("./test", "005746_2_2.jpg")
# a.show()
