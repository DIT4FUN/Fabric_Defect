import train.imgTool as imgTool

imgPath="./trainData/ori1/20181024"
#读取文件路径
pathL=imgTool.readIMGInDir(imgPath)
for id,filePath in enumerate(pathL):
    #中心剪裁
    imgTool.imgCentreCut(filePath)
    if id %10==0:
        print(id,"/",len(pathL))
print("Done!")
