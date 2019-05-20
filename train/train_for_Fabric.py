import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image
import train.imgTool as imgTool
import train.labelTool as labelTool

# 参数表
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
IMG_CUT = True  # 图像是否预处理


# 指定路径
path = './'
modelPath = path + "model/fabric.model"
imgPath = path + "trainData/ori1/20181024"
img_cutPath = path + "trainData/cutIMG"
labelPath = path + 'trainData/ori1/20181024_label'
print("模型文件夹路径" + modelPath)
print("原始图片文件夹路径" + imgPath)

# 参数初始化
place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)

# 加载数据
datatype = 'float32'
labelL = labelTool.readLabel(labelPath)  # 标签数据


def dataReader():
    def reader():
        # 图片预处理
        if IMG_CUT is True:
            imgpathL = imgTool.readIMGInDir(imgPath)
            for id, filePath in enumerate(imgpathL):
                # 中心剪裁
                imgTool.imgCentreCut(filePath, savePath=img_cutPath, detection=True)
                if id % 100 == 0:
                    print('|中心裁剪', id, "/", len(imgpathL))
        imgpathL=imgTool.readIMGInDir(img_cutPath)
        imgnameL =imgTool.readIMGInDir(img_cutPath,onle_name=True)
        for i in imgpathL:
            im=Image.open(i).convert("L")
            (H,W)=im.size
            im = np.array(im).reshape(1, H, W).astype(np.float32)


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
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])

    drop = fluid.layers.dropout(x=conv5, dropout_prob=0.5)
    fc1 = fluid.layers.fc(input=drop, size=512, act=None)
    bn = fluid.layers.batch_norm(input=fc1, act='relu')
    drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.5)
    fc2 = fluid.layers.fc(input=drop2, size=512, act=None)
    predict = fluid.layers.fc(input=fc2, size=10, act='softmax')
    return predict


# 新建项目
final_program = fluid.Program()
startup = fluid.Program()  # 默认启动程序

# 编辑项目
with fluid.program_guard(main_program=final_program, startup_program=startup):
    x_f = fluid.layers.data(name="x_f", shape=[1, 30, 15], dtype='float32')
    label_f = fluid.layers.data(name="label_f", shape=[1], dtype="int64")
    net_x_Base_f, pltdata = convolutional_neural_network2(x_f, "final_program")  # 获取网络
    # 定义损失函数
    cost_Base_f = fluid.layers.cross_entropy(input=net_x_Base_f, label=label_f)
    avg_cost_Base_f = fluid.layers.mean(cost_Base_f)
    final_programT = final_program.clone(for_test=True)
    # 定义优化方法
    sgd_optimizer_f = fluid.optimizer.Adam(learning_rate=0.01)
    sgd_optimizer_f.minimize(avg_cost_Base_f)
# 数据传入设置

# 人工标签传入
prebatch_reader = paddle.batch(
    reader=preDataReader(),
    batch_size=512)
prefeeder = fluid.DataFeeder(place=place, feed_list=[x, label])
batch_reader = paddle.batch(
    reader=dataReader(),
    batch_size=2048)
feeder = fluid.DataFeeder(place=place, feed_list=[x, label])

# 准备训练
exe.run(startup)

# 开始训练
for train_num, i in enumerate(range(TRAINNUM)):
    for batch_id, data in enumerate(prebatch_reader()):
        # 获取训练数据
        outs = exe.run(program=torNNBase,
                       feed=prefeeder.feed(data),
                       fetch_list=[label, net_x_Base, avg_cost_Base])
        # 按格式记录数据
        label_data, net_x_Base_data, avg_cost_Base_data = outs
        for i in range(len(label_data)):
            index = label_data[i].tolist()[0]
            basedata[index].append(net_x_Base_data[i].tolist())
