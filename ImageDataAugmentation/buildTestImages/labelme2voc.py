import os
import numpy as np
import codecs
import json
from glob import glob
import cv2
import shutil
from sklearn.model_selection import train_test_split
#http://spytensor.com/index.php/archives/35/?wolerm=g5q0b2
# 1.标签路径
#rootPath = "E:/ubuntushare/data/warehousetools/"
#labelme_path = rootPath + "original/"  # 原始labelme标注数据路径
#saved_path = rootPath + "VOC2007/"  # 保存路径

def labelme2vocFormat(labelmePath, vocPath):
    # 2.创建要求文件夹
    if not os.path.exists(vocPath + "Annotations"):
        os.makedirs(vocPath + "Annotations")
    if not os.path.exists(vocPath + "JPEGImages/"):
        os.makedirs(vocPath + "JPEGImages/")
    if not os.path.exists(vocPath + "ImageSets/Main/"):
        os.makedirs(vocPath + "ImageSets/Main/")

    # 3.获取待处理文件
    files = glob(labelmePath + "*.json")
    files = [i.replace("\\","/").split("/")[-1].split(".json")[0] for i in files]

    # 4.读取标注信息并写入 xml
    for json_file_ in files:
        json_filename = labelmePath + json_file_ + ".json"
        json_file = json.load(open(json_filename, "r", encoding="utf-8"))
        height, width, channels = cv2.imread(labelmePath + json_file_ + ".png").shape
        with codecs.open(vocPath + "Annotations/" + json_file_ + ".xml", "w", "utf-8") as xml:
            xml.write('<annotation>\n')
            xml.write('\t<folder>' + 'UAV_data' + '</folder>\n')
            xml.write('\t<filename>' + json_file_ + ".png" + '</filename>\n')
            xml.write('\t<source>\n')
            xml.write('\t\t<database>The UAV autolanding</database>\n')
            xml.write('\t\t<annotation>UAV AutoLanding</annotation>\n')
            xml.write('\t\t<image>flickr</image>\n')
            xml.write('\t\t<flickrid>NULL</flickrid>\n')
            xml.write('\t</source>\n')
            xml.write('\t<owner>\n')
            xml.write('\t\t<flickrid>NULL</flickrid>\n')
            xml.write('\t\t<name>ChaojieZhu</name>\n')
            xml.write('\t</owner>\n')
            xml.write('\t<size>\n')
            xml.write('\t\t<width>' + str(width) + '</width>\n')
            xml.write('\t\t<height>' + str(height) + '</height>\n')
            xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
            xml.write('\t</size>\n')
            xml.write('\t\t<segmented>0</segmented>\n')
            for multi in json_file["shapes"]:
                points = np.array(multi["points"])
                xmin = min(points[:, 0])
                xmax = max(points[:, 0])
                ymin = min(points[:, 1])
                ymax = max(points[:, 1])
                label = multi["label"]
                if xmax <= xmin:
                    pass
                elif ymax <= ymin:
                    pass
                else:
                    xml.write('\t<object>\n')
                    xml.write('\t\t<name>' + multi['label'] + '</name>\n')
                    xml.write('\t\t<pose>Unspecified</pose>\n')
                    xml.write('\t\t<truncated>1</truncated>\n')
                    xml.write('\t\t<difficult>0</difficult>\n')
                    xml.write('\t\t<bndbox>\n')
                    xml.write('\t\t\t<xmin>' + str(xmin) + '</xmin>\n')
                    xml.write('\t\t\t<ymin>' + str(ymin) + '</ymin>\n')
                    xml.write('\t\t\t<xmax>' + str(xmax) + '</xmax>\n')
                    xml.write('\t\t\t<ymax>' + str(ymax) + '</ymax>\n')
                    xml.write('\t\t</bndbox>\n')
                    xml.write('\t</object>\n')
                    print(json_filename, xmin, ymin, xmax, ymax, label)
            xml.write('</annotation>')

    # 5.复制图片到 VOC2007/JPEGImages/下
    image_files = glob(labelmePath + "*.png")
    print("copy image files to VOC007/JPEGImages/")
    for image in image_files:
        shutil.copy(image, vocPath + "JPEGImages/")

    # 6.split files for txt
    txtsavepath = vocPath + "ImageSets/Main/"
    ftrainval = open(txtsavepath + '/trainval.txt', 'w')
    ftest = open(txtsavepath + '/test.txt', 'w')
    ftrain = open(txtsavepath + '/train.txt', 'w')
    fval = open(txtsavepath + '/val.txt', 'w')

    total_files = glob(vocPath + "Annotations/*.xml")
    total_files = [i.replace("\\","/").split("/")[-1].split(".xml")[0] for i in total_files]
    # test_filepath = ""
    for file in total_files:
        ftrainval.write(file + "\n")
    # test
    # for file in os.listdir(test_filepath):
    #    ftest.write(file.split(".jpg")[0] + "\n")
    # split
    train_files, val_files = train_test_split(total_files, test_size=0.2, random_state=2)
    # train
    for file in train_files:
        ftrain.write(file + "\n")
    # val
    for file in val_files:
        fval.write(file + "\n")

    ftrainval.close()
    ftrain.close()
    fval.close()