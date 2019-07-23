from imgTool import readIMGInDir
from imgTool import drawIMG
from osTools import mkdir


# a = drawIMG("./testData", "222443_1_5Y.jpg")
def drawIMGL(path, savePath=None):
    """
    批量图像绘制
    :param path: 图片文件夹路径
    :return:
    """
    if savePath is None:
        mkdir(path + "/draw", de=True)
        savePath = path + "/draw"
    else:
        mkdir(savePath, de=True)

    imgNameL = readIMGInDir(path, onle_name=True)
    print("Find", len(imgNameL), "imgs")
    for imgName in imgNameL:
        im = drawIMG(path, imgName,quickMode=True)
        im.save(savePath + "/" + imgName)


# if __name__ == '__main__':
    # import argparse
    #
    # parser = argparse.ArgumentParser(description='请键入命令 --help 获取帮助')
    # parser.add_argument('--p', type=str)
    # parser.add_argument('--s', type=str)
    # args = parser.parse_args()
    # drawIMGL("./test")
path = input("请输入需要可视化图片所在的目录_")
drawIMGL(path)

input("按任意键结束")