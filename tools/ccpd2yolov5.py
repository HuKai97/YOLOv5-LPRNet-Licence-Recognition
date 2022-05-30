"""
@Author: HuKai
@Date: 2022/5/29  10:47
@github: https://github.com/HuKai97
"""
import shutil
import cv2
import os

def txt_translate(path, txt_path):
    for filename in os.listdir(path):
        print(filename)

        list1 = filename.split("-", 3)  # 第一次分割，以减号'-'做分割
        subname = list1[2]
        list2 = filename.split(".", 1)
        subname1 = list2[1]
        if subname1 == 'txt':
            continue
        lt, rb = subname.split("_", 1)  # 第二次分割，以下划线'_'做分割
        lx, ly = lt.split("&", 1)
        rx, ry = rb.split("&", 1)
        width = int(rx) - int(lx)
        height = int(ry) - int(ly)  # bounding box的宽和高
        cx = float(lx) + width / 2
        cy = float(ly) + height / 2  # bounding box中心点

        img = cv2.imread(path + filename)
        if img is None:  # 自动删除失效图片（下载过程有的图片会存在无法读取的情况）
            os.remove(os.path.join(path, filename))
            continue
        width = width / img.shape[1]
        height = height / img.shape[0]
        cx = cx / img.shape[1]
        cy = cy / img.shape[0]

        txtname = filename.split(".", 1)
        txtfile = txt_path + txtname[0] + ".txt"
        # 绿牌是第0类，蓝牌是第1类
        with open(txtfile, "w") as f:
            f.write(str(0) + " " + str(cx) + " " + str(cy) + " " + str(width) + " " + str(height))


if __name__ == '__main__':
    # det图片存储地址
    trainDir = r"K:\MyProject\datasets\ccpd\new\ccpd_2019\images\train\\"
    validDir = r"K:\MyProject\datasets\ccpd\new\ccpd_2019\images\val\\"
    testDir = r"K:\MyProject\datasets\ccpd\new\ccpd_2019\images\test\\"
    # det txt存储地址
    train_txt_path = r"K:\MyProject\datasets\ccpd\new\ccpd_2019\labels\train\\"
    val_txt_path = r"K:\MyProject\datasets\ccpd\new\ccpd_2019\labels\val\\"
    test_txt_path = r"K:\MyProject\datasets\ccpd\new\ccpd_2019\labels\test\\"
    txt_translate(trainDir, train_txt_path)
    txt_translate(validDir, val_txt_path)
    txt_translate(testDir, test_txt_path)