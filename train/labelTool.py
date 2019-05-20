import os

label1Key = {'N': '尼丝纺',
             'T': '塔丝隆',
             'P': '春亚纺',
             'S': '桃皮绒',
             'J': '锦涤纺',
             'R': '麂皮绒',
             'D': '涤塔夫',
             'Q': '其它品种'}

label2Key = {'T': '平纹',
             'W': '斜纹',
             'B': '格子',
             'S': '缎纹'}


def getLabelID(labelKey, debug=False):
    '''

    :param labelKey: 标签字典
    :return: 序号+字典Value的字典
    '''
    labelID = [(id, i[1]) for id, i in enumerate(labelKey.items())]
    labelID = dict(labelID)
    if debug is True:
        print(labelID)
    return labelID


def readTXTInDir(path, type=None):
    '''
    读取文件夹下所有文件的文件名和路径
    :param path: 路径
    type:指定文件类型，如果没有指定则视为jpg类型
    :return: 文件名列表
    '''
    if type is None:
        type = '.txt'
    nameL = []  # 保存文件名
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] == type:
                nameL.append(str(file).replace("\\", "/"))
    return nameL


def readLabel(labelPath, debug=False):
    '''
    一次性将所有图像标签加入内存
    :param imgPath: 图片目录
    :param labelPath: Label路径
    :return:标签字典{ID:Label}
    '''
    labelL=[]
    txtxfileL=readTXTInDir(labelPath)
    for i in txtxfileL:
        id=i[:6]
        with open(labelPath+"/"+i, "r") as f:
            info = f.readlines()
            label = info[1].replace("\n", "")[1:3]
            labelL.append((id,label))
    labelL=dict(labelL)
    if debug is True:
        print(labelL)
    return labelL

