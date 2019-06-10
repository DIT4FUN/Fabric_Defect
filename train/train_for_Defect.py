import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt
from torNN import TorNN
from imgTool import readIMGInDir as readIMGInDir
from imgTool import imgdetection as imgdetection
from osTools import mkdirL as mkdirL
from osTools import readDirName as readDirName
from pylab import mpl
from fabricNet import FabricNet

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 解决PIL显示乱码问题

# 参数表
# place = fluid.CUDAPlace(0)
place = fluid.CPUPlace()
TRAIN_DATA_SHAPE = (3, 1200, 2448)
C, IMG_H, IMG_W = TRAIN_DATA_SHAPE  # 输入网络图像大小
TRAINNUM = 100  # 训练次数
READIMG = 50  # 每次读取图片数量

# 指定路径
# 路径除root外均不带"/"后缀
path = './'
baseModelPath = path + "model/defectBase.model"
imgPath = path + "trainData/Classified2/classify"

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
        数据读取器
        目录结构为：./ 数字标签1 /所有该标签的图片
                  ./ 数字标签2 /所有该标签的图片
                  ...
        :return:
        '''
        for label in dirL:
            imgFilePathL = readIMGInDir(imgPath + "/" + str(label) + "/")
            for imgFilePath in imgFilePathL:
                # img_obj = imgdetection(imgFilePath)
                # im = img_obj.three2one()
                # im = Image.fromarray(cv.cvtColor(im, cv.COLOR_BGR2RGB))
                # im = im.resize((IMG_H, IMG_W), Image.ANTIALIAS)
                # im.show()
                im = Image.open(imgFilePath)
                im = np.array(im).reshape(3, IMG_W, IMG_H).astype(np.float32)
                # im = im / 255.0 * 2.0 - 1.0
                yield im, int(label)

    return reader


# 新建项目
defectProgram = fluid.Program()
startup = fluid.Program()  # 默认启动程序

# 编辑项目
with fluid.program_guard(main_program=defectProgram, startup_program=startup):
    x_f = fluid.layers.data(name="x_f", shape=[C, IMG_H, IMG_W], dtype='float32')
    label_f = fluid.layers.data(name="label_f", shape=[1], dtype="int64")
    net = FabricNet(class_dim=10)
    net_x = net.net(x_f)
    # 定义损失函数
    cost_Base_f = fluid.layers.cross_entropy(input=net_x, label=label_f)
    avg_cost_Base_f = fluid.layers.mean(fluid.layers.abs(cost_Base_f))
    acc = fluid.layers.accuracy(input=net_x, label=label_f, k=1)
    # final_programT = final_program.clone(for_test=True)
    # 定义优化方法
    sgd_optimizer_f = fluid.optimizer.Adam(learning_rate=0.01)
    sgd_optimizer_f.minimize(avg_cost_Base_f)

# 数据传入设置

prebatch_reader = paddle.batch(
    reader=paddle.reader.shuffle(dataReader(), 100),
    batch_size=READIMG)
prefeeder = fluid.DataFeeder(place=place, feed_list=[x_f, label_f])

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
                       fetch_list=[label_f, avg_cost_Base_f, acc, net_x])
        print(train_num, outs[1], outs[2])
        try:
            accsum.append(int(outs[2]))
            losssum.append(int(outs[1]))
        except:
            continue
        with open("./log.txt", "w") as f:
            f.writelines(str(train_num) + str(outs[1]) + "\n")
        accMean = sum(accsum) / len(accsum)
        lossMean = sum(losssum) / len(losssum)
        if accMean >= 0.7 and len(accsum) >= 6 or train_num == 20:
            fluid.io.save_inference_model(baseModelPath + str(batch_id) + str(accMean), ['x_f'], [net_x], exe,
                                          main_program=defectProgram)

    plt.figure(1)
    plt.title('布料识别指标-损失')
    plt.xlabel('迭代次数')
    plt.plot(train_num, lossMean)
    plt.figure(2)
    plt.plot(train_num, accMean)
    plt.title('布料识别指标-正确率')
    plt.xlabel('迭代次数')
    plt.show()
