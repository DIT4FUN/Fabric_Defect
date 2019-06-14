from PIL import Image

boxsixe = [240, 240]  # 测试框
input_img = [1200, 2448]  # 输入图片
vbox_input_img = [960, 2160]  # 虚拟图片大小

# 颜色对照表
L2color = {0: 0,38:1,75:2,113:3,14:4,52:5,89:6, }


def cut_box(img, mask):
    # 切割图片
    img = Image.open(img).convert('L')
    img = img.crop((144, 120, vbox_input_img[1] + 144, vbox_input_img[0] + 120))  # W H
    mask = Image.open(mask).convert('L')
    mask = mask.crop((144, 120, vbox_input_img[1] + 144, vbox_input_img[0] + 120))

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
                box = (list_W * long, list_H * long, (list_W + 1) * long, (list_H + 1) * long)
                mini_img = input_img.crop(box)
                mini_imgL.append(mini_img)
        return mini_imgL

    img_mini_imgL = box_creat(img)
    mask_mini_imgL = box_creat(mask)

    for id, mini_img in enumerate(mask_mini_imgL):
        colors = mini_img.getcolors()
        print(id, colors)
    # mask.show()


cut_box("color.png",
        "color.png")
