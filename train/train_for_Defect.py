import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt
from train.torNN import TorNN
from train.imgTool import readIMGInDir as readIMGInDir
from train.imgTool import imgdetection as imgdetection
from train.osTools import mkdirL as mkdirL
from train.net import net as netV2
from train.osTools import readDirName as readDirName
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 解决PIL显示乱码问题

# 参数表
# place = fluid.CUDAPlace(0)
place = fluid.CPUPlace()
IMG_H = 323  # 输入网络图像大小1958 960
IMG_W = 180
TRAINNUM = 50  # 训练次数
READIMG = 80  # 每次读取图片数量

# 指定路径
# 路径除root外均不带"/"后缀
path = './'
baseModelPath = path + "model/defectBase.model"
imgPath = path + "trainData/Classified"

labelFilePath = path + 'trainData/Classified/label.txt'
print("模型文件夹路径" + baseModelPath)
print("原始图片文件夹路径" + imgPath)

# 参数初始化
exe = fluid.Executor(place)

# 加载数据
dirL = readDirName(imgPath)  # 分类图片路径数据


def dataReader():
    def reader():
        '''
        目录结构为：./ 数字标签1 /所有该标签的图片
                  ./ 数字标签2 /所有该标签的图片
                  ...
        :return:
        '''
        for label in dirL:
            imgFilePathL = readIMGInDir(imgPath + "/" + str(label) + "/")
            for imgFilePath in imgFilePathL:
                img_obj = imgdetection(imgFilePath)
                im = img_obj.three2one()
                im = Image.fromarray(cv.cvtColor(im, cv.COLOR_BGR2RGB))
                im = im.resize((IMG_H, IMG_W), Image.ANTIALIAS)
                # im.show()
                im = np.array(im).reshape(3, IMG_W, IMG_H).astype(np.float32)
                # im = im / 255.0 * 2.0 - 1.0
                yield im, int(label)

    return reader


def convolutional_neural_network(img):
    # 第一个卷积-池化层
    # 使用20个5*5的滤波器，池化大小为2，池化步长为2，激活函数为Relu
    conv1 = fluid.layers.conv2d(input=img,
                                num_filters=32,
                                filter_size=3,
                                padding=1,
                                stride=1,
                                act='relu')

    pool1 = fluid.layers.pool2d(input=conv1,
                                pool_size=2,
                                pool_stride=2,
                                pool_type='max')

    bn1 = fluid.layers.batch_norm(input=pool1)
    fc1 = fluid.layers.fc(input=bn1, size=1024, act="relu")
    fc2 = fluid.layers.fc(input=fc1, size=512, act="relu")
    fc3 = fluid.layers.fc(input=fc2, size=256, act="relu")
    fc4 = fluid.layers.fc(input=fc3, size=128, act="relu")
    fc5 = fluid.layers.fc(input=fc4, size=7, act='softmax')
    return fc5


# 新建项目
defectProgram = fluid.Program()
startup = fluid.Program()  # 默认启动程序

# 编辑项目
with fluid.program_guard(main_program=defectProgram, startup_program=startup):
    x_f = fluid.layers.data(name="x_f", shape=[3, IMG_H, IMG_W], dtype='float32')
    label_f = fluid.layers.data(name="label_f", shape=[1], dtype="int64")
    # net_x = convolutional_neural_network(x_f)
    net_x = netV2(x_f, class_dim=7)
    # 定义损失函数
    cost_Base_f = fluid.layers.cross_entropy(input=net_x, label=label_f)
    avg_cost_Base_f = fluid.layers.mean(fluid.layers.abs(cost_Base_f))
    acc = fluid.layers.accuracy(input=net_x, label=label_f, k=1)
    # final_programT = final_program.clone(for_test=True)
    # 定义优化方法
    sgd_optimizer_f = fluid.optimizer.Adam(learning_rate=0.001)
    sgd_optimizer_f.minimize(avg_cost_Base_f)

# 数据传入设置

prebatch_reader = paddle.batch(
    reader=paddle.reader.shuffle(dataReader(), 50),
    batch_size=READIMG)
prefeeder = fluid.DataFeeder(place=place, feed_list=[x_f, label_f])

# 准备训练
exe.run(startup)
# 开始训练
for train_num, i in enumerate(range(TRAINNUM)):
    for batch_id, data in enumerate(prebatch_reader()):
        # print("Data Ready!")
        # 获取训练数据
        outs = exe.run(program=defectProgram,
                       feed=prefeeder.feed(data),
                       fetch_list=[label_f, avg_cost_Base_f, acc])
        print(train_num, outs[1], outs[2])
