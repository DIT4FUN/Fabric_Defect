import os
import imgTool

path = "./trainData/20181024"

# 切换工作目录
os.chdir(path)

imgFileL = imgTool.readIMGInDir("./", type="json", onle_name=True)
print(imgFileL)
# print(os.popen("dir").read())
print("Start")
for imgJson in imgFileL:
    cmd="labelme_json_to_dataset "+imgJson
    a=os.popen(cmd)
    #print(a.read())
print("Done")
