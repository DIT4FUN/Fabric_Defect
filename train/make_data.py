from PIL import Image
import imgTool
import osTools
import random
import shutil

boxsixe = [240, 240]  # 测试框
input_img = [1200, 2448]  # 输入图片
vbox_input_img = [960, 2160]  # 虚拟图片大小

# 颜色对照表
L2color = {0: 0, 38: 1, 75: 2, 113: 3, 14: 4, 52: 5, 89: 6, 57: 7, 128: 8, 19: 9, 94: 10}
L2text = {0: "无", 1: "布匹外", 2: "正常", 3: "油污", 4: "浆斑", 5: "停车痕", 6: "糙纬", 7: "笔迹", 8: "横向", 9: "纵向",
          10: "Other"}


def filter_img(pil_obj, mask_pil_obj, box_size=(120, 1200), step_size=60):
    """
    过滤非布匹区域，返回切割后仅含有布匹的图像
    :param mask_pil_obj: 蒙版图片的pil对象
    :param step_size: 滑动窗口步长
    :param box_size: 滑动窗口大小
    :param pil_obj: 原始图片的pil对象
    :return: 带瑕疵pil对象列表，完全正常Pillow列表
    """

    w, h = pil_obj.size
    block_w_num = (w // box_size[0]) * (box_size[0] // step_size) - 1
    pil_list1 = []
    pil_list2 = []
    pil_list3 = []
    pil_list_true = []
    pix_sum = box_size[0] * box_size[1]
    flag = True  # 只提取纯正常图片
    for id_w in range(block_w_num):
        box = (int(id_w * step_size), 0, int(id_w * step_size) + box_size[0], box_size[1])
        mini_img = pil_obj.crop(box)
        mini_img_mask = mask_pil_obj.crop(box)
        colors = mini_img_mask.getcolors()
        try:
            color_tag = [tag[1] for tag in colors]
        except IndexError:
            continue
        colors_d = dict(((v, k) for (k, v) in colors))
        if 0 in color_tag and 75 in color_tag:
            colors_d[75] += colors_d[0]

        if 38 not in color_tag:
            # 布匹外区域过滤

            if 75 in color_tag:
                # 瑕疵过滤 pix_sum越小越严格
                if 52 in color_tag and colors_d[75] / pix_sum >= 0.05:
                    # 过滤停车痕
                    pil_list1.append(mini_img)
                    flag = False
                if ((113 in color_tag or 14 in color_tag or 128 in color_tag or 89 in color_tag) and colors_d[75]
                        / pix_sum >= 0.8):
                    # 过滤 油污、浆斑、糙纬、横向
                    pil_list2.append(mini_img)
                    flag = False
                if 19 in color_tag and colors_d[75] / pix_sum >= 0.8:
                    pil_list3.append(mini_img)
                    flag = False
            if (75 in color_tag and 0 in color_tag and len(
                    color_tag) == 2 and block_w_num * 0.45 < id_w < block_w_num * 0.55 and flag is True):
                # 添加完全正常部分
                pil_list_true.append(mini_img)

    return pil_list1, pil_list2, pil_list3, pil_list_true


def first_cut_box(path, save_path):
    img_list = imgTool.read_img_in_dir(path, ext="png", name_none_ext=True)[0]

    def save_img(pil_list, path_):
        if len(pil_list) >= 1:
            for id_, img in enumerate(pil_list):
                img.save(path_ + str(file_name) + str(id_) + ".jpg")

    for file_name in img_list:
        mask_img = Image.open("./trainData/20181024/out/SegmentationClassPNG/" + str(file_name) + ".png").convert('L')
        ori_img = Image.open("./trainData/20181024/out/JPEGImages/" + str(file_name) + ".jpg").convert('L')
        all_pil_list = filter_img(ori_img, mask_img)
        for type_id, one_pil_list in enumerate(all_pil_list):
            save_img(one_pil_list, save_path + "/" + str(type_id))


# first_cut_box("./trainData/20181024/out/SegmentationClassPNG", "./data/cut")


def cut_box(img, mask, savePath, ext=0.5, extup=0.):
    '''

    :param img: 图像路径
    :param mask: 蒙版路径
    :param ext: 扩充偏移量[0-0.5] 不同的偏移量会带来更多的数据集
    :param extup:上下偏移量[0.3-1]
    :return:
    '''
    img_name = img[img.rindex("/"):-4] + str(ext * 10)

    # 切割图片
    extupL = 144 * (1 - extup)
    img = Image.open(img).convert('L')
    img = img.crop((extupL, 120, vbox_input_img[1] + extupL, vbox_input_img[0] + 120))  # W H
    mask = Image.open(mask).convert('L')
    mask = mask.crop((extupL, 120, vbox_input_img[1] + extupL, vbox_input_img[0] + 120))

    # print(mask.size) #2160 960
    def box_creat(input_img):
        long = boxsixe[0] // 2  # 步长
        '''

        :param input_img: PIL对象
        :return: 小块图像列表
        '''
        mini_imgL = []
        box_W = (2 * vbox_input_img[1]) // boxsixe[0]
        box_H = (2 * vbox_input_img[0]) // boxsixe[0]
        # print(box_W, box_H) 18 8
        for list_W in range(box_W):
            for list_H in range(box_H):
                if list_W % 2 == 1:
                    if list_W == box_W - 1:
                        continue
                    box = (list_W * long + ext * boxsixe[0], list_H * long, (list_W + 1) * long + ext * boxsixe[0],
                           (list_H + 1) * long)

                else:
                    box = (list_W * long, list_H * long, (list_W + 1) * long, (list_H + 1) * long)
                mini_img = input_img.crop(box)
                info = (str(list_W) + "-" + str(list_H), mini_img)
                mini_imgL.append(info)
        mini_imgL = dict(mini_imgL)  # W-H,PIL_obj
        return mini_imgL

    img_mini_imgL = box_creat(img)
    mask_mini_imgL = box_creat(mask)

    pix_sum = boxsixe[0] * boxsixe[1] * 0.25  # 14400
    count = 0  # 计数器
    for id, mini_img in mask_mini_imgL.items():
        colors = mini_img.getcolors()
        color_num = len(colors)
        colors = sorted(colors, key=lambda x: x[0], reverse=True)
        colorsL = [i for i in colors[0]]

        max_num = 0.5  # 默认最大阈值
        if color_num == 2:
            max_num = 0.5
        if color_num > 2:
            max_num = 0.7

        for num, color in colors:
            if 0 in colorsL:
                continue

            if num / pix_sum >= max_num:
                # 边缘情况
                if color == 75 and (38 in colorsL):
                    img_mini_imgL[id].save(savePath + "/" + str(img_name) + "-" + str(id) + "-10.jpg")
                    continue
                if 75 in colorsL:
                    max_num = 0.1
                label = L2color[color]
                img_mini_imgL[id].save(savePath + "/" + str(img_name) + "-" + str(id) + "-" + str(label) + ".jpg")
                count += 1
        # print(colors)
    return count


# cut_box("142518_3_7.jpg",
#         "142518_3_7.png", savePath="./outSave")
def make_data(path, savePath, ext=0.5, extup=0., de=True):
    '''

    :param path: labelme生产输出文件夹目录
    :param savePath: 保存目录
    :param ext: 扩充偏移量[0-0.5] 不同的偏移量会带来更多的数据集
    :param de:是否删除旧数据
    :return:
    '''
    osTools.mkdir(savePath, de=de)
    # imgL=imgTool.readIMGInDir(path+"/JPEGImages")
    maskL = imgTool.readIMGInDir(path + "/SegmentationClassPNG", type="png")

    count = 0
    for maskFile in maskL:
        imgFile = maskFile.replace("SegmentationClassPNG", "JPEGImages").replace("png", "jpg")
        try:
            num = cut_box(imgFile, maskFile, savePath, ext, extup)
        except:
            continue
        count += num
    print("Find", count, "datas --Done!")


# make_data('F:/Fabric_Defect2/train/trainData/20181024/out', 'F:/Fabric_Defect2/train/trainData/traindatas04', ext=0.0,
#           extup=0)


def clean_east_rain():
    fileL = imgTool.readIMGInDir('F:/Fabric_Defect2/train/trainData/20181024', type="json")
    for i in fileL:
        with open(i, "r") as f:
            info = f.read()
        info = info.replace("..\\\\20181024\\\\20181024\\\\", "")
        with open(i, "w")as f:
            f.write(info)


# clean_east_rain()

def random_data(path, save_path, buffer_size=4):
    name_list, file_list = imgTool.read_img_in_dir(path)
    end_num = len(file_list)
    random_list = []

    for i in range(0, end_num, buffer_size):
        if i // buffer_size == end_num // buffer_size - 1:
            break
        num = random.randint(i, i + buffer_size - 1)
        random_list.append(num)
        shutil.copyfile(file_list[num], save_path + "/test/" + name_list[num])

    for i in range(end_num):
        if i not in random_list:
            shutil.copyfile(file_list[i], save_path + "/train/" + name_list[i])


random_data("./data/cut", "./data")
