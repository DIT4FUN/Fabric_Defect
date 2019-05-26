import train.labelTool as labelTool
import train.imgTool as imgTool
import os


labelL = labelTool.readLabel("F:/布匹数据集/R/label")
imgname=imgTool.readIMGInDir("F:/布匹数据集/R/img",onle_name=True)
imgL=imgTool.readIMGInDir("F:/布匹数据集/R/img")

sum=0
for id,i in enumerate(imgname):
    try:
        label=labelL[i[:6]]
    except:
        os.remove(imgL[id])
        sum+=1
print(sum,len(imgname))


