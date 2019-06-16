from pylab import mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.sans-serif'] = ['SimHei']

with open("traindatalog299.txt", "r") as f:
    info = f.read().split("]")
    acc1 = info[0][1:].split(",")
    acc5 = info[1][2:].split(",")
    t_acc1 = info[2][2:].split(",")
    t_acc5 = info[3][2:].split(",")
    step = range(100)


    def str2float(L):
        FL = []
        for i in L:
            i = float(i[:5])
            FL.append(i)
        return FL


    acc1 = str2float(acc1)[:100]
    acc5 = str2float(acc5)[:100]
    t_acc1 = [0]+str2float(t_acc1)[:99]
    t_acc5 = str2float(t_acc5)[:100]
    plt.figure(1)
    plt.title('训练集瑕疵正确率识别指标')
    plt.xlabel('迭代次数')
    plt.ylabel('正确率')
    plt.plot(step, acc1, "r", label="Top1")
    plt.plot(step, acc5, "b", label="Top5")
    plt.legend(bbox_to_anchor=[1.05, 1])
    plt.grid()
    plt.show()
    plt.figure(2)
    plt.title('测试集瑕疵正确率识别指标')
    plt.xlabel('迭代次数')
    plt.ylabel('正确率')
    plt.plot(step, t_acc1, "r", label="Top1")
    plt.plot(step, t_acc5, "b", label="Top5")
    plt.legend(bbox_to_anchor=[1.05, 1])

    plt.grid()
    plt.show()

# str(accL1) + "," + str(accL5) + "," + str(t_accL1) + "," + str(t_accL5) + "," + str(costL) + "," + str(t_costL))
