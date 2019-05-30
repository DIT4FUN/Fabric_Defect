import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image
import imgTool as imgTool
import labelTool as labelTool
import matplotlib.pyplot as plt
from pylab import mpl
from net import net as netV2

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来显示中文

# 参数表
place = fluid.CUDAPlace(0)
# place = fluid.CPUPlace()
IMG_CUT = False  # 图像是否预处理
IMG_H = 32  # 输入网络图像大小
IMG_W = 32
TRAINNUM = 5  # 训练次数
READIMG=64 #每次读取图片数量
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




# 新建项目
fabricProgram = fluid.Program()
startup = fluid.Program()  # 默认启动程序

# 编辑项目
with fluid.program_guard(main_program=fabricProgram, startup_program=startup):
    x_f = fluid.layers.data(name="x_f", shape=[1, IMG_H, IMG_W], dtype='float32')
    label_f = fluid.layers.data(name="label_f", shape=[1], dtype="int64")
    #net_x = vgg_bn_drop(x_f)  # 获取网络
    net_x =netV2(x_f,3)
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
    reader=paddle.reader.shuffle(dataReader(),50),
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
        outs = exe.run(program=fabricProgram,
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
fluid.io.save_inference_model(modelPath, [x_f.name], [net_x], exe, main_program=fabricProgram)