# 加载库
import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image
import numpy
from imgTool import cut_box_for_infer as cbfi
from imgTool import readIMGInDir
import time
from osTools import mkdir

# 参数
img_size = (1, 1200, 2448)
box_size = (120, 120)  # 预选框尺寸
L2text = {0: "无", 1: "布匹外", 2: "正常", 3: "油污", 4: "浆斑", 5: "停车痕", 6: "糙纬", 7: "边缘", 8: "横向", 9: "纵向", 10: "Other"}

# 指定路径
path = "./"
params_dirname = path + "model/defectBase49Test80"
print("训练后文件夹路径" + params_dirname)

# 需要传入的参数
gpu_infer = False  # 是否使用GPU预测
quick_mode = True  # 是否使用快速模式
imgs_path = "./testData"  # 图片路径
save_path = imgs_path + "/info"  # 位置信息保存路径
mkdir(save_path,de=True)


def dataLReader(img_filePath):
    '''
    批量图片预处理工具
    :param img_filePathL: 图片所在目录
    :return: imgFinalL [(文件名:PIL对象列表),(文件名:PIL对象列表)...]
    '''
    imgFinalL = []
    img_filePathL = readIMGInDir(img_filePath)
    img_fileNameL = readIMGInDir(img_filePath, onle_name=True)

    for img_filePath, img_fileName in zip(img_filePathL, img_fileNameL):
        im = Image.open(img_filePath).convert('L')
        imgFinalL.append((img_fileName, cbfi(im, quick=quick_mode)))
    return imgFinalL


# 参数初始化
cpu = fluid.CUDAPlace(0)
exe = fluid.Executor(cpu)
prog = fluid.default_startup_program()
exe.run(prog)

# 加载模型
[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(params_dirname, exe)

imgFinalL = dataLReader(imgs_path)

# 读取文件

for img_name, img_PIL in imgFinalL:
    # 读取序列
    infoL=[]
    for id, data in enumerate(img_PIL):
        #start_time = time.time()
        im = numpy.array(data).reshape(1, 1, box_size[0], box_size[1]).astype(numpy.float32)
        results = exe.run(inference_program,
                          feed={feed_target_names[0]: im},
                          fetch_list=fetch_targets)
        end_time = time.time()
        #print("Time:", end_time - start_time)
        lab = np.argsort(results)[0][0][-1]
        lab2=np.argsort(results)[0][0][-2]

        # 序列 18 8
        if quick_mode is False:
            H = id % 8 + 1
            W = id // 8 + 1
        else:
            H = id % 4 + 1
            W = id // 4 + 1
        info=str(W)+"-"+str(H)+"-"+str(L2text[lab])+"-"+str(L2text[lab2])
        infoL.append(info)
    with open(save_path+"/"+str(img_name)+".txt","w") as f:
        for i in infoL:
            f.writelines(i+"\n")
        # data.save("./output/"+str(img_name)+"-"+str(id)+"L"+str(L2text[lab])+"W-H"+str(W)+str(H)+".jpg")

        # print(img_name, lab+1, "W-H", W, H)
