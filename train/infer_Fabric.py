try:
    import paddle.fluid as fluid
    import numpy as np
    from PIL import Image
    import cv2 as cv
    from imgTool import imgdetection as imgdetection
    from labelTool import translateLabel as tlabel
    import traceback
except:
    print("缺少模块，请检查Python包配置环境后重启该程序")
    exit("发现异常，退出程序")

print("程序载入ing... [当前预测环境--CPU预测模式]")
# 参数表
place = fluid.CPUPlace()  # CPU预测
IMG_H = 323  # 输入网络图像大小1958 960
IMG_W = 180
TRAINNUM = 50  # 训练次数
READIMG = 155

# 指定路径
# 路径除root外均不带"/"后缀
path = './'
testModelPath = path + "model/fabric.model"

# 参数初始化
exe = fluid.Executor(place)


def readIMG(imgFilePath):

    img_obj = imgdetection(imgFilePath)

    im = img_obj.three2one()
    im0 = Image.fromarray(cv.cvtColor(im, cv.COLOR_BGR2RGB))
    im0 = im0.resize((IMG_H, IMG_W), Image.ANTIALIAS)
    im0.save("./temp2.jpg")
    # im.show()
    im = np.array(im0).reshape(1, 3, IMG_W, IMG_H).astype(np.float32)
    # im = im / 255.0 * 2.0 - 1.0
    return im


[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=testModelPath,
                                                                                      executor=exe)
imgpath = input("模型准备完毕，请输入图片位置(请勿使用中文目录)__").replace("\\", "/")
print("图片路径为：" + imgpath)
try:
    img = readIMG(imgpath)
except:
    print(traceback.format_exc())
    print("请检查图片路径是否正确")
    exit("发现异常，退出程序")

print("图片读取成功，开始预测...")
try:
    results = exe.run(inference_program,
                      feed={feed_target_names[0]: img},
                      fetch_list=fetch_targets)
    lab = np.argsort(results)[0][0][-1]

    print(key[int(lab)])

except:
    print(traceback.format_exc())
    exit("请检查配置环境，Fluid启动失败")


input("运行结束")