import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image
import train.imgTool as imgTool
import train.labelTool as labelTool
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来显示中文，不然会乱码

# 参数表
place = fluid.CUDAPlace(0)
# place = fluid.CPUPlace()
IMG_CUT = False  # 图像是否预处理
IMG_H = 32  # 输入网络图像大小
IMG_W = 32
TRAINNUM = 20  # 训练次数
READIMG=256 #每次读取图片数量
# 指定路径
path = './'
modelPath = path + "model/fabric.model"
imgPath = path + "trainData/ori1/20181024"
img_cutPath = path + "trainData/cutIMG"
labelPath = path + 'trainData/ori1/20181024_label'
print("模型文件夹路径" + modelPath)
print("原始图片文件夹路径" + imgPath)

# 参数初始化

exe = fluid.Executor(place)

# 加载数据
datatype = 'float32'
labelL = labelTool.readLabel(labelPath)  # 标签数据
if IMG_CUT is True:
    imgpathL = imgTool.readIMGInDir(imgPath)
    for id, filePath in enumerate(imgpathL):
        # 中心剪裁
        imgTool.imgCentreCut(filePath, savePath=img_cutPath, detection=True)
        if id % 100 == 0:
            print('|中心裁剪', id, "/", len(imgpathL))


def dataReader():
    def reader():
        # 图片预处理
        trueLabelNum=0
        imgpathL = imgTool.readIMGInDir(img_cutPath, type="png")
        # imgnameL = [i[:6] for i in imgTool.readIMGInDir(img_cutPath, onle_name=True)]
        for id, i in enumerate(imgpathL):
            im = Image.open(i).convert("L")
            (H, W) = im.size
            im = np.array(im).reshape(1, H, W).astype(np.float32)
            im = im / 255.0 * 2.0 - 1.0
            imFileName = i[i.rindex("/") + 1:i.rindex("/") + 7]  # 获取文件名
            try:
                label = labelL[str(imFileName)]  # 得到Label
                trueLabelNum+=1
            except:
                continue
            yield im, label
        print("TrueIMG:",trueLabelNum)

    return reader


# 定义神经网络模型
def vgg_bn_drop(input):
    def conv_block(ipt, num_filter, groups, dropouts):
        return fluid.nets.img_conv_group(
            input=ipt,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act='relu',
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type='max')

    conv1 = conv_block(input, 64, 2, [0.3, 0])
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])


    drop = fluid.layers.dropout(x=conv3, dropout_prob=0.5)
    fc1 = fluid.layers.fc(input=drop, size=512, act=None)
    bn = fluid.layers.batch_norm(input=fc1, act='relu')
    drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.5)
    fc2 = fluid.layers.fc(input=drop2, size=512, act=None)
    predict = fluid.layers.fc(input=fc2, size=10, act='softmax')
    return predict


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
final_program = fluid.Program()
startup = fluid.Program()  # 默认启动程序

# 编辑项目
with fluid.program_guard(main_program=final_program, startup_program=startup):
    x_f = fluid.layers.data(name="x_f", shape=[1, IMG_H, IMG_W], dtype='float32')
    label_f = fluid.layers.data(name="label_f", shape=[1], dtype="int64")
    #net_x = vgg_bn_drop(x_f)  # 获取网络
    net_x =convolutional_neural_network(x_f)
    # 定义损失函数
    cost_Base_f = fluid.layers.cross_entropy(input=net_x, label=label_f)
    avg_cost_Base_f = fluid.layers.mean(fluid.layers.abs(cost_Base_f))
    acc=fluid.layers.accuracy(input=net_x, label=label_f, k=1)
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
PLT_train_num=[]
PLT_cost=[]
PLT_acc=[]
for train_num, i in enumerate(range(TRAINNUM)):
    T = True  # 每个循环近输出第一次结果
    for batch_id, data in enumerate(prebatch_reader()):

        # 获取训练数据
        outs = exe.run(program=final_program,
                       feed=prefeeder.feed(data),
                       fetch_list=[label_f, avg_cost_Base_f,acc])
        if T is True:
            PLT_train_num.append(train_num)
            PLT_cost.append(outs[1][0])
            PLT_acc.append(outs[2][0])
            print(train_num, outs[1],outs[2])
            T=False


plt.figure(1)
plt.title('布料识别指标-损失')
plt.xlabel('迭代次数')
plt.plot(PLT_train_num, PLT_cost)
plt.figure(2)
plt.plot(PLT_train_num, PLT_acc)
plt.title('布料识别指标-正确率')
plt.xlabel('迭代次数')
plt.show()
fluid.io.save_inference_model(modelPath, [x_f.name], [net_x], exe,main_program=final_program)