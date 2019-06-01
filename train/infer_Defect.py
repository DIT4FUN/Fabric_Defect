#调整工作目录
import os
os.chdir('F:/Fabric_Defect2/train')


try:
    import paddle.fluid as fluid
    import numpy as np
    from PIL import Image
    import cv2 as cv
    from imgTool import imgdetection as imgdetection
    from imgTool import readIMGInDir
    import traceback
except:
    print(traceback.format_exc())
    print("缺少模块，请检查Python包配置环境后重启该程序")
    temp=input("按任意键退出")
    exit("发现异常，退出程序")

print("程序载入ing... [当前预测环境--CPU预测模式]")
# 参数表
place = fluid.CPUPlace()  # CPU预测
IMG_H = 323  # 输入网络图像大小1958 960
IMG_W = 180
TRAINNUM = 50  # 训练次数
READIMG = 155

key = {0: '正常',
       1: '油污',
       2: '浆斑',
       3: '糙纬',
       4: '停车痕',
       5: '横线的',
       6: '竖线的'}
# 指定路径
# 路径除root外均不带"/"后缀
path = './'
testModelPath = path + "model/defectBase.model48INF0.86754966"

# 参数初始化
exe = fluid.Executor(place)


def readIMG(imgFilePath):

    img_obj = imgdetection(imgFilePath)
    #imOri=Image.open(imgFilePath)
    #imOri=imOri.resize((IMG_H, IMG_W), Image.ANTIALIAS)
    #imOri.save("./temp1.jpg")

    im = img_obj.three2one()
    im0 = Image.fromarray(cv.cvtColor(im, cv.COLOR_BGR2RGB))
    im0 = im0.resize((IMG_H, IMG_W), Image.ANTIALIAS)
    # im.show()
    im = np.array(im0).reshape(1, 3, IMG_W, IMG_H).astype(np.float32)
    # im = im / 255.0 * 2.0 - 1.0
    return im,im0

try:
    [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=testModelPath,
                                                                                      executor=exe)
except:
    print("无法读取模型文件")
    temp = input("按任意键退出")
    exit("发现异常，退出程序")
imgpath = input("模型准备完毕，请输入图片位置(请勿使用中文目录)__").replace("\\", "/")
print("图片路径为：" + imgpath)

files=readIMGInDir(imgpath)
print("图片数量为：",len(files))
imgL=[]
try:
    for file in files:
        img,imF = readIMG(file)
        imgL.append(img)
except:
    print(traceback.format_exc())
    print("请检查图片路径是否正确")
    temp=input("按任意键退出")
    exit("发现异常，退出程序")

print("图片读取成功，开始预测...")
try:
    for img in imgL:
        results = exe.run(inference_program,
                          feed={feed_target_names[0]: img},
                          fetch_list=fetch_targets)
        lab = np.argsort(results)[0][0][-1]

        print(key[int(lab)])

except:
    print(traceback.format_exc())
    temp=input("按任意键退出")
    exit("请检查配置环境，Fluid启动失败")
print("渲染图片ing...")

#imF.show()

input("运行结束")
