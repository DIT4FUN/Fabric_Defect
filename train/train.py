import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter
from paddle.fluid.initializer import init_on_cpu
from pylab import mpl
from se_resnext import SE_ResNeXt50_32x4d


mpl.rcParams['font.sans-serif'] = ['SimHei']  # 解决PIL显示乱码问题

# 参数表
# place = fluid.CUDAPlace(0)#GPU训练
place = fluid.CPUPlace()  # CPU训练
TRAIN_DATA_SHAPE = (1, 30, 300)
C, IMG_H, IMG_W = TRAIN_DATA_SHAPE  # 输入网络图像大小
TRAINNUM = 1000  # 训练次数
READIMG = 100  # 每次读取图片数量

class_dim = 10  # 分类总数
LEARNING_RATE = 0.0005  # 学习率

# 指定路径
# 路径除root外均不带"/"后缀
path = './'
baseModelPath = path + "model/defectBase"  # 模型保存路径
data_path = path + "data"
train_imgPath = data_path + "/train"  # 训练集路径
test_imgPath = data_path + "/test"  # 测试集路径

print("模型文件夹路径" + baseModelPath)

# 参数初始化
exe = fluid.Executor(place)

# 新建项目
defectProgram = fluid.Program()  # 主程序

startup = fluid.Program()  # 默认启动程序

# 编辑项目
with fluid.program_guard(main_program=defectProgram, startup_program=startup):
    image = fluid.layers.data(name="image", shape=[C, IMG_H, IMG_W], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    net_x = SE_ResNeXt50_32x4d().net(input=image, class_dim=class_dim)
    # 定义损失函数
    cost = fluid.layers.cross_entropy(net_x, label)
    avg_cost = fluid.layers.mean(cost)
    # 获取正确率
    acc_1 = fluid.layers.accuracy(input=net_x, label=label, k=1)
    acc_5 = fluid.layers.accuracy(input=net_x, label=label, k=5)
    # 动态测试程序
    testProgram = defectProgram.clone(for_test=True)
    # 定义优化方法
    adma_optimizer = fluid.optimizer.Adam(learning_rate=LEARNING_RATE)
    adma_optimizer.minimize(avg_cost)

# 数据传入设置

train_reader = paddle.batch(
    reader=paddle.reader.shuffle(dataReaderA(train_imgPath), 3000),
    batch_size=READIMG)
train_feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

test_reader = paddle.batch(
    reader=paddle.reader.shuffle(dataReaderA(test_imgPath), 1000),
    batch_size=READIMG)
test_feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

# 准备训练
exe.run(startup)
# 开始训练
accL1 = []
t_accL1 = []
accL5 = []
t_accL5 = []
costL = []
t_costL = []

maxAcc = 0
t_maxAcc = 0

for train_num, i in enumerate(range(TRAINNUM)):
    sumacc1 = []
    t_sumacc1 = []
    sumacc5 = []
    t_sumacc5 = []
    sumcost = []
    t_sumcost = []

    for batch_id, data in enumerate(train_reader()):
        # 获取训练数据
        outs = exe.run(program=defectProgram,
                       feed=train_feeder.feed(data),
                       fetch_list=[acc_1, acc_5, cost])
        print("Train step:", train_num, batch_id, "Acc", outs[0], outs[1], 'Cost:', sum(outs[2]) / len(outs[2]))
        try:
            sumacc1.append(float(outs[0]))
            sumacc5.append(float(outs[1]))
            sumcost.append(sum(outs[2]) / len(outs[2]))
        except:
            pass
    for batch_id, data in enumerate(test_reader()):
        t_outs = exe.run(program=testProgram,
                         feed=test_feeder.feed(data),
                         fetch_list=[acc_1, acc_5, cost])
        print("Test: Acc", t_outs[0], t_outs[1], 'Cost:', sum(t_outs[2]) / len(t_outs[2]))
        try:
            t_sumacc1.append(float(t_outs[0]))
            t_sumacc5.append(float(t_outs[1]))
            t_sumcost.append(sum(t_outs[2]) / len(t_outs[2]))
        except:
            pass
    if len(sumacc1) == 0 or len(t_sumacc1) == 0 or len(sumacc5) == 0 or len(t_sumacc5) == 0 or len(
            t_sumcost) == 0 or len(sumcost) == 0:
        continue
    avgacc = sum(sumacc1) / len(sumacc1)
    avgacc5 = sum(sumacc5) / len(sumacc5)
    accL1.append(avgacc)
    accL5.append(avgacc5)

    t_avgacc = sum(t_sumacc1) / len(t_sumacc1)
    t_avgacc5 = sum(t_sumacc5) / len(t_sumacc5)
    t_accL1.append(t_avgacc)
    t_accL5.append(t_avgacc5)

    tcost = sum(sumcost) / len(sumcost)
    costL.append(tcost)
    t_costL.append(sum(t_sumcost) / len(t_sumcost))
    print(train_num, "Acc", avgacc, avgacc5, "TAcc", t_avgacc, t_avgacc5, tcost)

    if avgacc > maxAcc:
        maxAcc = avgacc
        fluid.io.save_inference_model(dirname=baseModelPath + str(train_num) + "Train" + str(int(maxAcc * 100)),
                                      feeded_var_names=["image"], target_vars=[net_x], main_program=defectProgram,
                                      executor=exe)
    if t_avgacc > t_maxAcc:
        t_maxAcc = t_avgacc
        fluid.io.save_inference_model(dirname=baseModelPath + str(train_num) + "Test" + str(int(t_maxAcc * 100)),
                                      feeded_var_names=["image"], target_vars=[net_x], main_program=defectProgram,
                                      executor=exe)
    if train_num % 100 == 99:
        fluid.io.save_persistables(dirname=baseModelPath + str(train_num) + "persistables", executor=exe,
                                   main_program=defectProgram)
with open(path + "traindatalog.txt", "w") as f:
    f.writelines(
        str(accL1) + "," + str(accL5) + "," + str(t_accL1) + "," + str(t_accL5) + "," + str(costL) + "," + str(t_costL))
