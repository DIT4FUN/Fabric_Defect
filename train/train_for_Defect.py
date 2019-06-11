import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt
from fabricReader import train as dataReader
from fabricNet import FabricNet
from pylab import mpl
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter
from paddle.fluid.initializer import init_on_cpu

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 解决PIL显示乱码问题

# 参数表
# place = fluid.CUDAPlace(0)
place = fluid.CPUPlace()
TRAIN_DATA_SHAPE = (3, 1200, 2448)
C, IMG_H, IMG_W = TRAIN_DATA_SHAPE  # 输入网络图像大小
TRAINNUM = 100  # 训练次数
READIMG = 3  # 每次读取图片数量
class_dim = 10
LEARNING_RATE = 0.003  # 学习率
POWER = 0.9  # 学习率控制
TOTAL_STEP = 10  # 训练轮数

# 指定路径
# 路径除root外均不带"/"后缀
path = './'
baseModelPath = path + "model/defectBase.model"
imgPath = path + "trainData/Classified2/classify"

labelFilePath = path + 'trainData/Classified/label.txt'
print("模型文件夹路径" + baseModelPath)
print("原始图片文件夹路径" + imgPath)


def poly_decay():
    global_step = _decay_step_counter()
    with init_on_cpu():
        decayed_lr = LEARNING_RATE * (fluid.layers.pow(
            (1 - global_step / TOTAL_STEP), POWER))
    return decayed_lr


# 参数初始化
exe = fluid.Executor(place)

# 加载数据

# 新建项目
defectProgram = fluid.Program()
startup = fluid.Program()  # 默认启动程序

# 编辑项目
with fluid.program_guard(main_program=defectProgram, startup_program=startup):
    no_grad_set = []


    def create_loss(predict, label ,mask):
        print(predict.shape, label.shape)
        predict = fluid.layers.transpose(predict, perm=[0, 2, 3, 1])
        predict = fluid.layers.reshape(predict, shape=[-1, class_dim])
        label = fluid.layers.reshape(label, shape=[-1, 1])
        predict = fluid.layers.gather(predict, mask)
        label = fluid.layers.gather(label, mask)
        label = fluid.layers.cast(label, dtype="int64")

        print(predict.shape, label.shape)
        loss = fluid.layers.softmax_with_cross_entropy(predict, label)
        no_grad_set.append(label.name)
        return fluid.layers.reduce_mean(loss)


    image = fluid.layers.data(name="x_f", shape=[C, IMG_H, IMG_W], dtype='float32')
    label_sub = fluid.layers.data(name='label_sub1', shape=[1], dtype='int32')
    mask_sub1 = fluid.layers.data(name='mask_sub1', shape=[-1], dtype='int32')
    net = FabricNet(class_dim=class_dim)
    net_x = net.net(image)  # out_F (-1, 10, 300, 612)
    # 定义损失函数
    cost_Base_f = create_loss(net_x, label_sub,mask_sub1)

    # final_programT = final_program.clone(for_test=True)
    # 定义优化方法
    regularizer = fluid.regularizer.L2Decay(0.0001)  # 权重衰减 防止过拟合
    optimizer = fluid.optimizer.Momentum(
        learning_rate=poly_decay(), momentum=0.9, regularization=regularizer)
    _, params_grads = optimizer.minimize(cost_Base_f, no_grad_set=no_grad_set)

# 数据传入设置

prebatch_reader = paddle.batch(
    reader=paddle.reader.shuffle(dataReader(), 50),
    batch_size=READIMG)
prefeeder = fluid.DataFeeder(place=place, feed_list=[image, label_sub,mask_sub1])

# 准备训练
exe.run(startup)
# 开始训练
for train_num, i in enumerate(range(TRAINNUM)):
    accsum = []
    losssum = []
    accMean = 0
    lossMean = 0
    for batch_id, data in enumerate(prebatch_reader()):
        # print("Data Ready!")
        # 获取训练数据
        outs = exe.run(program=defectProgram,
                       feed=prefeeder.feed(data),
                       fetch_list=[cost_Base_f])
        print(train_num, outs[0])
