import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来显示中文，不然会乱码

id=[]
loss=[]
acc=[]
with open('./model/log.txt',"r") as f:
    info=f.readlines()
    for i in info:
        i=i.replace("\n",'').split(" ")
        id.append(i[0])
        loss.append(float(i[1][1:-1]))
        acc.append(float(i[2][1:-1]))
plt.figure(1)
plt.title('瑕疵识别指标-损失')
plt.xlabel('迭代次数')
plt.plot(id, loss)
plt.figure(2)
plt.plot(id, acc)
plt.title('瑕疵识别指标-准确率')
plt.xlabel('迭代次数')
plt.show()

