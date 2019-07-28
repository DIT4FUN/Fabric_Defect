from imgTool import read_img_in_dir
import os


for i in range(6):
    img_name, img_list = read_img_in_dir("./data2/train/" + str(i), dir_deep=1)
    for id_, img_path in enumerate(img_list):
        new_name = str(i) + str(id_) + img_name[id_] + ".jpg"
        os.rename(img_path, img_path[:img_path.rindex("/")]+new_name)
