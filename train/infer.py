import time
import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image
from imgTool import ImgPretreatment, cut_edge
from osTools import mkdir

# Hyper parameter
use_cuda = True  # Whether to use GPU or not
batch_size = 10  # Number of incoming batches of data
model_path = "./model/first_model"  # infer model path
img_file_path = "./test/220425_1_3Y.jpg"

mkdir("./temp")


# Reader
def data_reader(img_file_path):
    # 单张1.09 s ,10张2.31 s ,净单张0.122 s
    return_list = []
    img = Image.open(img_file_path).convert('L')
    temp_dir = "./temp/" + str(time.time())[:9]
    mkdir(temp_dir, de=True)
    img = cut_edge(img)
    for index, im in enumerate(img):
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
    return_list = list(return_list)*10
    return return_list


# Initialization

place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
startup = fluid.Program()
exe.run(startup)

# Feed configure
# if you want to shuffle "reader=paddle.reader.shuffle(dataReader(), buf_size)"
img_list = data_reader(img_file_path)

# if you want to asynchronous reading
# batch_reader = fluid.io.PyReader(feed_list=[x, y], capacity=64)
# batch_reader.decorate_sample_list_generator(paddle.batch(data_reader(), batch_size=batch_size),place)

# load infer model
[infer_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(model_path, exe)

# Start infer


s = time.time()
for id_, data in enumerate(img_list):
    results = exe.run(infer_program, feed={feed_target_names[0]: data}, fetch_list=fetch_targets)
    lab = np.argsort(results)[0][0][-1]
    # print(lab)
print(time.time() - s)
