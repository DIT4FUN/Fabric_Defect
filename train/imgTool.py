'''
ImgTool
处理图片工具集合
'''
import os
import sys
import random
import traceback

from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import numpy as np

fontP = "./font/1.ttf"


def readIMGInDir(path, type=None, onle_name=False):
    """
    读取文件夹下所有文件的文件名和路径
    :param path: 路径
    :param type:指定文件类型，如果没有指定则视为jpg类型
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


def read_img_in_dir(path, dir_deep=0, ext=None, name_none_ext=False):
    """
    读取文件夹下所有文件的文件名和路径
    :param path: 路径
    :param dir_deep:文件夹检索深度，默认为0
    :param ext:指定文件类型，如果没有指定则视为jpg类型
    :param name_none_ext:如果为True则返回的文件名列表中不含有扩展名
    :return: nameL:文件夹内所有路径+文件名 './trainData/ori1/20181024/000030_1_0.jpg' , '000030_1_0.jpg' or '000030_1_0'
    """
    if ext is None:
        ext = '.jpg'
    else:
        ext = "." + ext
    name_list = []  # 保存文件名
    name_path = []
    for id_, (root, dirs, files) in enumerate(os.walk(path)):
        for file in files:
            if os.path.splitext(file)[1] == ext:
                if name_none_ext is True:
                    name_list.append(os.path.splitext(file)[0])
                else:
                    name_list.append(file)
                name_path.append(str(os.path.join(root, file)).replace("\\", "/"))
        if id_ == dir_deep:
            break
    return name_list, name_path


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
            final_img_pil_obj = tool_img_pretreatment.req_img()

    Tips:
    1、第一次进行图像减颜色均值操作时过程较长，并非卡顿。
    2、如果要统一数据集尺寸，则最好放在所有对图像的操作之前，初始化图像操作之后。这样更有利运行速度。
    参数保存:
    在获取颜色均值后会保一个参数文件'img_pretreatment.txt'记录参数。
    参数读取仅在for_test=True或ignore_log=False时生效，
    """

    def __init__(self, all_img_path, mean_color_num=500, dir_deep=0, img_type="jpg", read_img_type='L',
                 ignore_log=False, for_test=False, debug=True):
        """
        :param all_img_path: 图像文件所在文件夹路径
        :param mean_color_num:颜色均值采样数
        :param dir_deep:检索文件夹的深度
        :param img_type: 图像文件类型，默认jpg格式
        :param read_img_type:图像读取通道类型 默认为灰度
        :param ignore_log:是否忽略之前参数文件
        :param for_test:是否用于测试
        :param debug:设置为False后将进入哑巴模式，什么信息都不会打印在屏幕上，报错信息除外
        """
        if debug:
            print("----------ImgPretreatment Start!----------")

        self.img_file_name, self.img_files_path = read_img_in_dir(all_img_path, dir_deep, img_type, name_none_ext=True)

        self.len_img = len(self.img_files_path)
        self.mean_color_num = mean_color_num
        self.img_type = img_type
        self.read_img_type = read_img_type
        self.debug = debug
        self.shape = Image.open(self.img_files_path[0]).size

        # Flag 变量
        self.allow_req_img = False  # 图像初始化控制
        self.allow_save_flag = True  # 允许保存，如果进行去均值操作则不允许保存
        self.__need_color_cut_flag = False  # 是否需要颜色去均值
        self.__first_print_flag = True  # 是否第一次输出控制变量

        self.color_mean_flag = False
        if ignore_log is False or for_test is True:
            try:
                with open("./img_pretreatment.txt", "r") as f:
                    info = f.read().split("-")
                    check_info = str(self.read_img_type) + str(self.len_img) + str(self.mean_color_num)
                    if info[0] == check_info or for_test:
                        self.color_mean_flag = True
                        self.__color_mean = float(info[1][1:-1])
                        if debug:
                            print("Load Log Successfully!")
            except:
                assert not for_test, "Load Log Finally! Place "
        # 当前进程变量
        self.now_index = 1
        self.now_img_obj_list = []
        self.now_img_obj_list.append(Image.open(self.img_files_path[0]).convert(self.read_img_type))
        if debug:
            print("Data read successfully number:", self.len_img)

    def img_init(self, index):
        """
        图像初始化
        :param index: 需要初始化图像的索引
        """
        self.allow_save_flag = True

        if index is not self.now_index or len(self.now_img_obj_list) != 1:
            try:
                self.now_img_obj_list = []
                self.now_img_obj_list.append(Image.open(self.img_files_path[index]).convert(self.read_img_type))
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
        only_shape = None
        for id_, imgP in enumerate(self.img_files_path):
            im = Image.open(imgP).convert(self.read_img_type)
            if id_ == 0:
                only_shape = im.size
            if only_shape != im.size:
                mean_color_num += 1
                continue
            sum_img_numpy += np.mean(np.asarray(im).reshape((1, im.size[0], im.size[1])))
            success_num += 1
            if id_ == mean_color_num or id_ == self.len_img - 1:
                self.__color_mean = np.around((sum_img_numpy / success_num), decimals=3).tolist()
                self.color_mean_flag = True

                with open("./img_pretreatment.txt", "w") as f:
                    f.write(
                        (str(self.read_img_type) + str(self.len_img) + str(self.mean_color_num) + "-" + str(
                            self.__color_mean)))
                    if self.debug is True:
                        print("Color mean :", self.__color_mean)
                    print("Successful counting in color mean:", success_num - 1)
                    print("Write log --Done!")
                return self.__color_mean

    def img_cut_color(self):
        """
        颜色去均值操作
        防止因为去均值后颜色模式发生改变，所以立了个Flag，使它即将结束时运行该操作。
        """
        self.__need_color_cut_flag = True
        self.allow_save_flag = False

    def img_only_one_shape(self, expect_h, expect_w):
        """
        传入图片将修正为数据集统一的尺寸
        """
        pass

    def img_resize(self, expect_w, expect_h):
        """
        多相位调整图像大小
        :param expect_w: 期望的宽度-横向
        :param expect_h: 期望的高度-纵向
        :return:
        """
        temp_list = []
        for now_img_obj in self.now_img_obj_list:
            img = now_img_obj.resize((expect_w, expect_h), Image.LANCZOS)
            temp_list.append(img)
        self.now_img_obj_list = temp_list
        self.shape = temp_list[0].size

    def random_crop(self):
        pass

    def img_rotate(self, angle_range=(0, 0), angle_step=1, angle_and_transpose=False, only_transpose=False):
        """
        图像翻转
        如果仅返回规则翻转，则不需要修改前两个参数
        :param angle_range:旋转最小角度和最大角度
        :param angle_step:选择角度之间的步长
        :param angle_and_transpose:旋转角度之后再进行水平和垂直旋转
        :param only_transpose:仅进行水平和垂直旋转
        Tips:如果使用该模块，则只在最后获取时运行
        """

        def tran(ori_pil_obj, out_pil_list):
            out_pil_list.append(ori_pil_obj.transpose(Image.FLIP_LEFT_RIGHT))
            out_pil_list.append(ori_pil_obj.transpose(Image.FLIP_TOP_BOTTOM))

        temp_list = list(self.now_img_obj_list)
        if only_transpose is True:
            for now_img_obj in self.now_img_obj_list:
                tran(now_img_obj, temp_list)
            self.now_img_obj_list = temp_list
        else:
            for now_img_obj in self.now_img_obj_list:
                for angle in range(angle_range[0], angle_range[1], angle_step):
                    temp_list.append(now_img_obj.rotate(angle))
            if angle_and_transpose is True:
                self.now_img_obj_list = []
                for now_img_obj in temp_list:
                    tran(now_img_obj, self.now_img_obj_list)

    def img_random_noise(self):
        """
        随机添加噪声
        """
        pass

    def img_random_vague(self):
        """
        随机模糊
        """
        pass

    def img_random_contrast(self, random_num=1, lower=0.5, upper=1.5):
        """
        随机对比度
        :param random_num: 随机次数，尽可能在3以内，建议为1，均匀随机
        :param lower:最低可能的对比度
        :param upper:最高可能的对比度
        """

        temp_list = list(self.now_img_obj_list)
        for seed in range(1, random_num + 1):
            factor = random.uniform(lower + ((upper - lower) * seed - 1 / random_num),
                                    lower + ((upper - lower) * seed / random_num))
            for now_img_obj in self.now_img_obj_list:
                img = ImageEnhance.Sharpness(now_img_obj)
                img = img.enhance(factor)
                temp_list.append(img)
        self.now_img_obj_list = temp_list

    def img_random_brightness(self, random_num=1, lower=0.5, upper=1.5):
        """
        随机亮度
        :param random_num: 随机次数，尽可能在3以内，建议为1，均匀随机
        :param lower:最低可能的亮度
        :param upper:最高可能的亮度
        """

        temp_list = list(self.now_img_obj_list)
        for seed in range(1, random_num + 1):
            factor = random.uniform(lower + ((upper - lower) * seed - 1 / random_num),
                                    lower + ((upper - lower) * seed / random_num))
            for now_img_obj in self.now_img_obj_list:
                img = ImageEnhance.Brightness(now_img_obj)
                img = img.enhance(factor)
                temp_list.append(img)
        self.now_img_obj_list = temp_list

    def img_random_saturation(self, random_num=1, lower=0.5, upper=1.5):
        """
        随机饱和度
        :param random_num: 随机次数，尽可能在3以内，建议为1，均匀随机
        :param lower:最低可能的亮度
        :param upper:最高可能的亮度
        """
        temp_list = list(self.now_img_obj_list)
        for seed in range(1, random_num + 1):
            factor = random.uniform(lower + ((upper - lower) * seed - 1 / random_num),
                                    lower + ((upper - lower) * seed / random_num))
            for now_img_obj in self.now_img_obj_list:
                img = ImageEnhance.Color(now_img_obj)
                img = img.enhance(factor)
                temp_list.append(img)
        self.now_img_obj_list = temp_list

    def req_img(self, save_path=None):
        """
        获取当前处理进程中图片
        :param save_path:如果保存图片，则需要提供保存路径
        :return:PIL_Obj or PIL_Obj_List
        """
        # 特殊操作区域
        if self.__need_color_cut_flag is True:
            temp_list = []
            for now_img_obj in self.now_img_obj_list:
                now_img_obj = Image.fromarray(np.asarray(now_img_obj) - self._req_color_mean())
                temp_list.append(now_img_obj)
            self.now_img_obj_list = list(temp_list)
            self.__need_color_cut_flag = False

        # 输出区域

        if self.debug and self.__first_print_flag:
            print("The current size of the first image output is ", self.shape)
            print("The number of single image pre-processed is expected to be ", len(self.now_img_obj_list))
            print("The total number of pictures expected to be produced is ", self.len_img * len(self.now_img_obj_list))
            self.__first_print_flag = False
        if self.debug:
            self.__progress_print()
        # 保存区域
        if save_path is not None:
            assert self.allow_save_flag, "Can not save F mode img! Please undo img_cut_color operation!"
            folder = os.path.exists(save_path)
            if not folder:
                os.makedirs(save_path)
            if len(self.now_img_obj_list) != 1:
                for id_, img in enumerate(self.now_img_obj_list):
                    img.save(
                        os.path.join(save_path, self.img_file_name[self.now_index] + str(id_) + ".jpg").replace("\\",
                                                                                                                "/"))
            else:
                self.now_img_obj_list[0].save(
                    os.path.join(save_path, self.img_file_name[self.now_index] + ".jpg").replace("\\", "/"))
        # 数据返回区域
        return self.now_img_obj_list

    def _req_color_mean(self):
        if self.color_mean_flag:
            return self.__color_mean
        else:
            return self.__color_mean_start()

    def __progress_print(self):
        """
        打印进度百分比
        """

        percentage = (self.now_index + 1) / self.len_img
        stdout_obj = sys.stdout
        style = "|\\|"
        if self.now_index % 2 == 0:
            style = "|/|"
        if self.now_index == self.len_img - 1:
            stdout_obj.write('\rPercentage of progress:{:.2%}'.format(percentage))
            print("\n----------ImgPretreatment Done!-----------\n")
        else:
            stdout_obj.write('\r' + style + '  Percentage of progress:{:.2%}'.format(percentage))

    def __len__(self):
        return self.len_img


# 测试代码
all_img_tool = ImgPretreatment(all_img_path="test/1", debug=True, ignore_log=True)
for i in range(all_img_tool.len_img):
    all_img_tool.img_init(i)
    # all_img_tool.img_rotate(only_transpose=True)
    # all_img_tool.img_random_brightness()
    # all_img_tool.img_random_contrast()
    # all_img_tool.img_cut_color()
    all_img_tool.img_resize(612, 300)
    # all_img_tool.img_random_saturation()

    all_img_tool.req_img(save_path="./test/save1")

    # all_img_tool.req_img()


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

# a = drawIMG("./test/", "005746_2_2.jpg")
# a.show()
