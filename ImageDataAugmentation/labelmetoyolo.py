# trans_labelme_to_yolo.py
import cv2
import os
import json
import numpy as np
from pathlib import Path
from glob import glob

cls2id = {'chipping': 0,'collapse':1,'slumping':2}

def labelme2yolo_single(img_path, label_file):
    anno = json.load(open(label_file, "r", encoding="utf-8"))
    shapes = anno['shapes']
    w0, h0 = anno['imageWidth'], anno['imageHeight']
    labels = []
    for s in shapes:
        pts = s['points']
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        x = (x1 + x2) / 2 / w0
        y = (y1 + y2) / 2 / h0
        w = abs(x2 - x1) / w0
        h = abs(y2 - y1) / h0
        cid = cls2id[s['label']]
        labels.append([cid, x, y, w, h])

    return np.array(labels)

def labelme2yolo(img_path, labelme_label_dir, save_dir='res/'):
    labelme_label_dir = str(Path(labelme_label_dir)) + '/'
    save_dir = str(Path(save_dir))
    yolo_label_dir = save_dir + '/'
    if not os.path.exists(yolo_label_dir):
        os.makedirs(yolo_label_dir)
    json_files = glob(labelme_label_dir + '*.json')

    for jf in json_files:
        filename = os.path.basename(jf).rsplit('.', 1)[0]
        labels = labelme2yolo_single(img_path, jf)
        if len(labels) > 0:
            np.savetxt(yolo_label_dir + filename + '.txt', labels)
    print('Completed!')

if __name__ == '__main__':
    img_path = 'E:/newpaper2024/Paper2/data/sample/pics/'
    json_dir = 'E:/newpaper2024/Paper2/data/sample/json/'
    save_dir = 'E:/newpaper2024/Paper2/data/sample/output/'
    labelme2yolo(img_path, json_dir, save_dir)
