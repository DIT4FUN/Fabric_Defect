import paddle.fluid as fluid
import paddle
import numpy as np
from resnet_vd import ResNet50_vd
from imgTool import ImgPretreatment

# Hyper parameter
# place = fluid.CUDAPlace(0)#GPU训练
place = fluid.CPUPlace()  # CPU训练
TRAIN_DATA_SHAPE = (1, 30, 30)
C, IMG_H, IMG_W = TRAIN_DATA_SHAPE  # 输入网络图像大小
TRAINNUM = 800  # 训练次数
READIMG = 1000  # 每次读取图片数量

class_dim = 30  # 分类总数
LEARNING_RATE = 0.0005  # 学习率

# 指定路径
# 路径除root外均不带"/"后缀
path = './'
base_model_path = path + "model/defectBase"  # 模型保存路径
data_path = path + "data2"
train_img_path = data_path + "/train"  # 训练集路径
test_img_path = data_path + "/test"  # 测试集路径
use_cuda = True  # Whether to use GPU or not

print("模型文件夹路径" + base_model_path)


# Reader
def data_reader(for_test=False):
    def reader():
        if for_test is False:
            img_tool = ImgPretreatment(train_img_path, mean_color_num=3000, dir_deep=1)
            for index in range(img_tool.len_img):
                img_tool.img_init(index)
                img_tool.img_only_one_shape(120, 120)
                img_tool.img_resize(30, 30)
                img_tool.img_random_saturation(2)
                img_tool.img_random_contrast(2)
                img_tool.img_random_brightness(2)
                img_tool.img_rotate(only_transpose=True)
                img_tool.img_cut_color()
                label = img_tool.now_img_name[0]
                img_l = img_tool.req_img()
                for img in img_l:
                    img = np.array(img).reshape(1, 30, 30).astype(np.float32)
                    yield img, label
        else:
            img_tool = ImgPretreatment(test_img_path, for_test=True)
            for index in range(img_tool.len_img):
                img_tool.img_init(index)
                img_tool.img_only_one_shape(120, 120)
                img_tool.img_resize(30, 30)
                img_tool.img_cut_color()
                label = img_tool.now_img_name[0]
                img_l = img_tool.req_img()
                for img in img_l:
                    img = np.array(img).reshape(1, 30, 30).astype(np.float32)
                    yield img, label

    return reader


# Initialization
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)

# Program
defect_program = fluid.Program()  # 主程序
startup = fluid.Program()  # 默认启动程序

# Edit Program
with fluid.program_guard(main_program=defect_program, startup_program=startup):
    image = fluid.layers.data(name="image", shape=[C, IMG_H, IMG_W], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    net_x = ResNet50_vd().net(input=image, class_dim=class_dim)
    net_x = fluid.layers.fc(net_x, 6, act="softmax")
    # 定义损失函数
    cost = fluid.layers.cross_entropy(net_x, label)
    avg_cost = fluid.layers.mean(cost)
    # 获取正确率
    acc_1 = fluid.layers.accuracy(input=net_x, label=label, k=1)
    acc_5 = fluid.layers.accuracy(input=net_x, label=label, k=5)
    # 动态测试程序
    testProgram = defect_program.clone(for_test=True)
    # 定义优化方法
    adma_optimizer = fluid.optimizer.Adam(learning_rate=LEARNING_RATE)
    adma_optimizer.minimize(avg_cost)

# 数据传入设置

train_reader = paddle.batch(
    reader=paddle.reader.shuffle(data_reader(for_test=False), 3000),
    batch_size=READIMG)
train_feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

test_reader = paddle.batch(
    reader=paddle.reader.shuffle(data_reader(for_test=True), 1000),
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
        outs = exe.run(program=defect_program,
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

    fluid.io.save_inference_model(dirname=base_model_path + str(train_num) + "Train" + str(
        int(avgacc * 100)) + "Test" + str(int(t_avgacc * 100)),
                                  feeded_var_names=["image"], target_vars=[net_x], main_program=defect_program,
                                  executor=exe)

    if train_num % 100 == 99:
        fluid.io.save_persistables(dirname=base_model_path + str(train_num) + "persistables", executor=exe,
                                   main_program=defect_program)
    with open(path + "traindatalog.txt", "a") as f:
        f.writelines(
            str(accL1) + "," + str(accL5) + "," + str(t_accL1) + "," + str(t_accL5) + "," + str(costL) + "," + str(
                t_costL))
