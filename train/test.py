path="./"

accL1 = []
t_accL1 = []
accL5 = []
t_accL5 = []
costL = []
t_costL = []

with open(path + "traindatalog.txt", "w") as f:
    f.writelines(
        str(accL1) + "," + str(accL5) + "," + str(t_accL1) + "," + str(t_accL5) + "," + str(costL) + "," + str(t_costL))