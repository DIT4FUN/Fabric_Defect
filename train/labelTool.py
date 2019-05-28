import os
import traceback

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


def getDict():
    '''
    获取当前Key
    :return:
    '''
    return label1Key, label2Key


def getLabelID(labelKey, debug=False, readID=False):
    '''

    :param labelKey: 标签字典
    readID:更换返回值
    :return: 序号+字典Value的字典 更换后为字典Value+序号
    '''
    labelID = [(id, i[0]) for id, i in enumerate(labelKey.items())]
    if readID is True:
        labelID = [(i[0], id) for id, i in enumerate(labelKey.items())]
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
    一次性将所有布匹类型标签加入内存
    :param labelPath: Label路径
    :return:标签字典{ID:Label}
    '''
    labelL = []
    txtxfileL = readTXTInDir(labelPath)
    key1 = getLabelID(label1Key, readID=True)
    key2 = getLabelID(label2Key, readID=True)
    for i in txtxfileL:
        id = i[:6]
        with open(labelPath + "/" + i, "r") as f:
            info = f.readlines()
            label = info[1].replace("\n", "")[1:3]
            try:
                labelL.append((id, int(key1[label[0]]) + int(key2[label[1]]) * 10))
            except:
                # print(traceback.format_exc())
                continue
    labelL = dict(labelL)
    if debug is True:
        print(labelL)
    return labelL


# readLabel('./trainData/ori1/20181024_label',debug=True)

def readclassify(labelPath, debug=False):
    '''
    一次性将所有图像分类标签加入内存
    :param labelPath: Label路径
    :return:标签字典{ID:Label}
    '''
    labelL = []
    txtxfileL = readTXTInDir(labelPath)
    for i in txtxfileL:
        id = i[:6]
        try:
            with open(labelPath + "/" + i, "r") as f:
                info = f.readlines()
                label = int(info[0].replace("\n", "")[-1])
                labelL.append((id,label))
        except:
                print(traceback.format_exc())
                continue
    labelL = dict(labelL)
    if debug is True:
        print(labelL)
    return labelL

#print(readclassify("F:/Fabric_Defect2/train/trainData/Classified2/label",debug=True))

def translateLabel(label):
    '''

    :param label: int标签
    :return: str标签
    '''
    key1 = getLabelID(label1Key)
    key2 = getLabelID(label2Key)
    strLabel = label1Key[key1[int(label) % 10]] + label2Key[key2[int(label) // 10]]

    return strLabel

# print(translateLabel(4))
