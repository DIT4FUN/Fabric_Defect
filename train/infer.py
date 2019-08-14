import time
import paddle.fluid as fluid
import numpy as np
from PIL import Image
from imgTool import ImgPretreatment, cut_edge
from osTools import mkdir

# Hyper parameter
use_cuda = True  # Whether to use GPU or not
batch_size = 10  # Number of incoming batches of data
model_path = "./model/f1_model"  # infer model path
model_path2 = "./model/f2_model"  # infer model path
img_file_path = "./test/220425_1_3Y.jpg"

mkdir("./temp")


# Reader
def f1_data_reader(img_file_path):
    """
    第一阶段预测读取函数
    该阶段将预测布匹大概情况
    :param img_file_path: 图像文件夹
    :return: 图像列表，原图候选框位置
    """
    # 单张1.09 10张2.31
    return_list = []
    img = Image.open(img_file_path).convert('L')
    temp_dir = "./temp/" + str(time.time())[:9]
    mkdir(temp_dir, de=True)
    img_list, box_list = cut_edge(img)
    for index, im in enumerate(img_list):
        im.save(temp_dir + "/" + str(index) + ".jpg")
    img_tool = ImgPretreatment(temp_dir, for_test=True)
    for index in range(img_tool.len_img):
        img_tool.img_init(index)
        img_tool.img_only_one_shape(600, 1200)
        img_tool.img_resize(150, 300)
        img_tool.img_cut_color()
        img_l = img_tool.req_img()
        for img in img_l:
            img = np.array(img).reshape(1, 1, 150, 300).astype(np.float32)
            return_list.append(img)
    # return_list = list(return_list)*10
    return return_list, box_list


def f2_data_reader(img, box, step=15):
    """
    第二阶段预测读取函数
    :param img: 单个图片文件
    :param box: 在原图中的位置
    :param step: 滑动窗口步长
    :return: 图像列表，原图候选框位置
    """
    img = img.reshape(150, 300)
    w = 150
    block_num = (w // 30) * (30 // step)-1
    return_list = []
    v_box_list = []
    for block in range(block_num):
        crop_box = (block * step, 0, block * step + 30, 300)
        v_box = (box[0] + block * step * 4, 0, box[0] + block * step * 4 + 120, 1200)
        im = img[crop_box[0]:crop_box[2], crop_box[1]:crop_box[3]]
        im = im.reshape(1, 1, 30, 300)
        return_list.append(im)
        v_box_list.append(v_box)
    return return_list, v_box_list


def f3_data_reader(box):
    pass


# Initialization

place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
startup = fluid.Program()

# Feed configure
# if you want to shuffle "reader=paddle.reader.shuffle(dataReader(), buf_size)"


# load infer model

f1_scope = fluid.Scope()
f2_scope = fluid.Scope()
se_scope = fluid.Scope()
with fluid.scope_guard(f1_scope):
    [infer_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(model_path, exe)
with fluid.scope_guard(f2_scope):
    [infer_program2, feed_target_names2, fetch_targets2] = fluid.io.load_inference_model(model_path2, exe)

f1_data_list = f1_data_reader(img_file_path)
# Start infer


exe.run(startup)

s = time.time()

f1_save_data = []
f1_save_box = []
with fluid.scope_guard(f1_scope):
    for id_, data in enumerate(f1_data_list[0]):
        results = exe.run(infer_program, feed={feed_target_names[0]: data}, fetch_list=[fetch_targets[0]])
        lab = np.argsort(results)[0][0][-1]
        if lab == 1:
            f1_save_data.append(data)
            f1_save_box.append(f1_data_list[1][id_])
        print(lab, f1_data_list[1][id_])

f2_save_data = []
f2_save_box = []
with fluid.scope_guard(f2_scope):
    for id2_ in range(len(f1_save_data)):
        f2_data_list = f2_data_reader(f1_save_data[id2_], f1_save_box[id2_])
        for id22_, data2 in enumerate(f2_data_list[0]):
            results2 = exe.run(infer_program2, feed={feed_target_names2[0]: data2}, fetch_list=[fetch_targets2[0]])
            lab2 = np.argsort(results2)[0][0][-1]
            f2_save_data.append(data2)
            print(lab2, f2_data_list[1][id22_])

print("time", time.time() - s)
