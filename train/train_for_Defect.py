import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image
import train.imgTool as imgTool
import train.labelTool as labelTool
import matplotlib.pyplot as plt
from train.torNN import TorNN
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来显示中文，不然会乱码

# 参数表
place = fluid.CUDAPlace(0)
# place = fluid.CPUPlace()
IMG_H = 64  # 输入网络图像大小
IMG_W = 64
TRAINNUM = 20  # 训练次数
READIMG = 256  # 每次读取图片数量

# 指定路径
# 路径除root外均不带"/"后缀
path = './'
modelPath = path + "model/defect.model"
imgPath = path + "trainData"
labelPath = path + 'trainData'
print("模型文件夹路径" + modelPath)
print("原始图片文件夹路径" + imgPath)

# 参数初始化

exe = fluid.Executor(place)


# 加载数据

def dataReader():
    def reader():
        pass

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

    fc2 = fluid.layers.fc(input=bn1, size=10, act='relu')
    pltdata = fluid.layers.fc(input=fc2, size=3, act=None)
    return fc2


# 新建项目
defectProgram = fluid.Program()
startup = fluid.Program()  # 默认启动程序

# 编辑项目
with fluid.program_guard(main_program=defectProgram, startup_program=startup):
    x_f = fluid.layers.data(name="x_f", shape=[1, IMG_H, IMG_W], dtype='float32')
    label_f = fluid.layers.data(name="label_f", shape=[1], dtype="int64")
    # net_x = vgg_bn_drop(x_f)  # 获取网络
    net_x = convolutional_neural_network(x_f)
    # 定义损失函数
    cost_Base_f = fluid.layers.cross_entropy(input=net_x, label=label_f)
    avg_cost_Base_f = fluid.layers.mean(fluid.layers.abs(cost_Base_f))
    acc = fluid.layers.accuracy(input=net_x, label=label_f, k=1)
    # final_programT = final_program.clone(for_test=True)
    # 定义优化方法
    sgd_optimizer_f = fluid.optimizer.SGD(learning_rate=0.01)
    sgd_optimizer_f.minimize(avg_cost_Base_f)

# 数据传入设置

prebatch_reader = paddle.batch(
    reader=dataReader(),
    batch_size=READIMG)
prefeeder = fluid.DataFeeder(place=place, feed_list=[x_f, label_f])

# 准备训练
exe.run(startup)

# 开始训练
