import train.imgTool as imgTool
import cv2 as cv

imgPath="./trainData/ori1/20181024"
#imgPath="./trainData/ori2"
#读取文件路径
pathL=imgTool.readIMGInDir(imgPath)
#处理图像
for id,filePath in enumerate(pathL):
    #中心剪裁
    img=imgTool.imgCentreCut(filePath,savePath='./trainData/centre',detection=True)

    if id %10==0:
        print(id,"/",len(pathL))
print("Done!")
