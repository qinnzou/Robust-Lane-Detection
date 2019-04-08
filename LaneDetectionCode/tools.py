import os
import shutil
import random
from PIL import Image

def flipImage(path1, path2, Format):
    fileList = os.listdir(path1)
    if not os.path.exists(path2):
        os.makedirs(path2)
    for item in fileList:
        fileName = os.path.join(path1, item)
        flipName = os.path.join(path2, item)
        if os.path.isfile(fileName) and fileName.endswith(Format):
            im = Image.open(fileName)
            im.transpose(Image.FLIP_LEFT_RIGHT).save(flipName)
            print("Saved: " + flipName)
        elif os.path.isdir(fileName):
            flipImage(fileName, flipName, Format)

def readIndex(index_fpath):
    data = []
    with open(index_fpath, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            data.append(lines)
    file_to_read.close()
    return data

def splitIndex(index_fpath, split_index_path, val_rate = 0.1, test_rate=0.0, is_random=False, op=op_split_lstm):
    data = readIndex(index_fpath)
    if is_random:
        random.shuffle(data)

    train_rate = 1 - val_rate - test_rate
    train_num = int(len(data) * train_rate)
    val_num = int(len(data) * val_rate)
    test_num = int(len(data) - train_num - val_num)
    with open(split_index_path+"train_index.txt", 'w') as train_file:
        for i in range(train_num):
            lines = op(data[i])
            train_file.write(lines)
            print("train_index: %s/%s"%(i+1,train_num))
    train_file.close()
    if val_rate != 0.0:
        with open(split_index_path + "val_index.txt", 'w') as val_file:
            for i in range(train_num,train_num+val_num):
                lines = op(data[i])
                val_file.write(lines)
                print("val_index: %s/%s" % (i + 1- train_num, val_num))
        val_file.close()
    if test_rate != 0.0:
        with open(split_index_path + "test_index.txt", 'w') as test_file:
            for i in range(train_num+val_num,train_num+val_num+test_num):
                lines = op(data[i])
                test_file.write(lines)
                print("test_index: %s/%s" % (i + 1 - train_num - val_num, test_num))
        test_file.close()

def rebuildFileFolder(path): #清空eval文件夹
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
