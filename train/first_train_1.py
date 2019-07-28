import paddle.fluid as fluid
import paddle
import numpy as np
from pylab import mpl
from resnet import ResNet34
from imgTool import ImgPretreatment

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 解决PIL显示乱码问题

# 参数表
place = fluid.CUDAPlace(0)  # GPU训练
C, IMG_H, IMG_W = 1, 150, 300
TRAIN_NUM = 1000  # 训练次数
READIMG = 100  # 每次读取图片数量
class_dim = 30  # 分类总数
LEARNING_RATE = 0.0005  # 学习率

# 指定路径
# 路径除root外均不带"/"后缀
path = './'
base_model_path = path + "model/defectBase"  # 模型保存路径
params_path = path + "model/ResNet34_pretrained"
data_path = path + "data"
train_img_path = data_path + "/train"  # 训练图片路径
test_img_path = data_path + "/test"  # 测试图片路径

print("模型文件夹路径" + base_model_path)


def data_reader(for_test=False):
    """
    数据读取函数
    :param for_test: 是否用于测试
    :return: 图片数据以及标签(0：正常，1：异常)
    """

    def reader():
        if for_test is False:
            img_tool = ImgPretreatment(train_img_path, mean_color_num=3000)
            for index in range(img_tool.len_img):
                img_tool.img_init(index)
                img_tool.img_only_one_shape(600, 1200)
                img_tool.img_resize(150, 300)
                img_tool.img_random_saturation(2)
                img_tool.img_random_contrast(2)
                img_tool.img_random_brightness(2)
                img_tool.img_rotate(only_transpose=True)
                img_tool.img_cut_color()
                tag = img_tool.now_img_name[0]
                img_l = img_tool.req_img()
                label = 1
                if int(tag) == 3:
                    label = 0
                for img in img_l:
                    img = np.array(img).reshape(1, 150, 300).astype(np.float32)
                    yield img, label
        else:
            img_tool = ImgPretreatment(test_img_path, for_test=True)
            for index in range(img_tool.len_img):
                img_tool.img_init(index)
                img_tool.img_only_one_shape(600, 1200)
                img_tool.img_resize(150, 300)
                img_tool.img_cut_color()
                tag = img_tool.now_img_name[0]
                img_l = img_tool.req_img()
                label = 1
                if int(tag) == 3:
                    label = 0
                for img in img_l:
                    img = np.array(img).reshape(1, 150, 300).astype(np.float32)
                    yield img, label

    return reader


# 参数初始化
exe = fluid.Executor(place)

# 新建项目
first_program = fluid.Program()  # 主程序
fluid.io.load_params(executor=exe, main_program=first_program, dirname=params_path)
startup = fluid.Program()  # 默认启动程序

# 编辑项目
with fluid.program_guard(main_program=first_program, startup_program=startup):
    image = fluid.layers.data(name="image", shape=[C, IMG_H, IMG_W], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    net_x = ResNet34().net(input=image, class_dim=class_dim)
    net_x =fluid.layers.fc(input=net_x,size=2,act="softmax")
    # 定义损失函数
    cost = fluid.layers.cross_entropy(net_x, label)
    avg_cost = fluid.layers.mean(cost)
    # 获取正确率
    acc_1 = fluid.layers.accuracy(input=net_x, label=label, k=1)
    # 动态测试程序
    testProgram = first_program.clone(for_test=True)
    # 定义优化方法
    adma_optimizer = fluid.optimizer.Adam(learning_rate=LEARNING_RATE)
    adma_optimizer.minimize(avg_cost)

# 数据传入设置

train_reader = paddle.batch(
    reader=paddle.reader.shuffle(data_reader(), 1000),
    batch_size=READIMG)
train_feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

test_reader = paddle.batch(
    reader=paddle.reader.shuffle(data_reader(for_test=True), 500),
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

for train_num, i in enumerate(range(TRAIN_NUM)):
    sumacc1 = []
    t_sumacc1 = []
    sumacc5 = []
    t_sumacc5 = []
    sumcost = []
    t_sumcost = []

    for batch_id, data in enumerate(train_reader()):
        # 获取训练数据
        outs = exe.run(program=first_program,
                       feed=train_feeder.feed(data),
                       fetch_list=[acc_1, acc_1, cost])
        # print("Train step:", train_num, batch_id, "Acc", outs[0], outs[1], 'Cost:', sum(outs[2]) / len(outs[2]))
        try:
            sumacc1.append(float(outs[0]))
            sumacc5.append(float(outs[1]))
            sumcost.append(sum(outs[2]) / len(outs[2]))
        except:
            pass
    for batch_id, data in enumerate(test_reader()):
        t_outs = exe.run(program=testProgram,
                         feed=test_feeder.feed(data),
                         fetch_list=[acc_1, acc_1, cost])
        # print("Test: Acc", t_outs[0], t_outs[1], 'Cost:', sum(t_outs[2]) / len(t_outs[2]))
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
                                  feeded_var_names=["image"], target_vars=[net_x], main_program=first_program,
                                  executor=exe)

    if train_num % 100 == 99:
        fluid.io.save_persistables(dirname=base_model_path + str(train_num) + "persistables", executor=exe,
                                   main_program=first_program)
    with open(path + "traindatalog.txt", "a") as f:
        f.writelines(
            str(accL1) + "," + str(accL5) + "," + str(t_accL1) + "," + str(t_accL5) + "," + str(costL) + "," + str(t_costL))
