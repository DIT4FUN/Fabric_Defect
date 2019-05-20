import os

def file_name(file_dir):

    for root, dirs, files in os.walk(file_dir):
        print(root)  # 当前目录路径
        print(dirs)  # 当前路径下所有子目录
        print(files)  # 当前路径下所有非目录子文件





def fileRead(path):
    #a=os.listdir("./trainData")
    filesL=[]
    for root, dirs, files in os.walk(path):
        filesL.append(files)
    print("|找到",len(filesL)-1,"文件")
    return filesL[1:]

