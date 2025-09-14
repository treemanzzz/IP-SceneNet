import os
import random
import shutil


def moveFile(fileDir, trainDir):
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    rate1 = 0.2
    picknumber1 = int(filenumber * rate1)  # 按照rate比例从文件夹中取一定数量的文件
    sample1 = random.sample(pathDir, picknumber1)  # 随机选取picknumber数量的样本
    print(sample1)
    print(len(sample1))
    for name in sample1:
        shutil.move(fileDir + '/' + name, trainDir)
    return


if __name__ == "__main__":
    fileDir = "/home/ExtraData/SceneClass/Data/water"
    testDir = "/home/ExtraData/SceneClass/Data/test_data/water"
    moveFile(fileDir, testDir)
