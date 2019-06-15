from PIL import Image
import imgTool
import osTools

boxsixe = [240, 240]  # 测试框
input_img = [1200, 2448]  # 输入图片
vbox_input_img = [960, 2160]  # 虚拟图片大小

# 颜色对照表
L2color = {0: 0, 38: 1, 75: 2, 113: 3, 14: 4, 52: 5, 89: 6, 57: 7, 128: 8, 19: 9, 94: 10}
L2text = {0: "无", 1: "布匹外", 2: "正常", 3: "油污", 4: "浆斑", 5: "停车痕", 6: "糙纬", 7: "笔迹", 8: "横向", 9: "纵向", 10: "Other"}


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
        colors = sorted(colors, key=lambda x: colors[0], reverse=True)
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


make_data('F:/Fabric_Defect2/train/trainData/20181024/out', 'F:/Fabric_Defect2/train/trainData/traindatas04', ext=0.0,
          extup=0)


def clean_east_rain():
    fileL = imgTool.readIMGInDir('F:/Fabric_Defect2/train/trainData/20181024', type="json")
    for i in fileL:
        with open(i, "r") as f:
            info = f.read()
        info = info.replace("..\\\\20181024\\\\20181024\\\\", "")
        with open(i, "w")as f:
            f.write(info)

# clean_east_rain()
