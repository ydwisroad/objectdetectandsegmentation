import math
import os
from copy import copy
from pathlib import Path
import random

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import re
from PIL import Image, ImageDraw, ImageFont
import platform
import json

def is_writeable(dir, test=False):
    # Return True if directory has write permissions, test opening a file with write permissions if test=True
    if test:  # method 1
        file = Path(dir) / 'tmp.txt'
        try:
            with open(file, 'w'):  # open file with write permissions
                pass
            file.unlink()  # remove file
            return True
        except IOError:
            return False
    else:  # method 2
        return os.access(dir, os.R_OK)  # possible issues on Windows

def user_config_dir(dir='Ultralytics', env_var='YOLOV5_CONFIG_DIR'):
    # Return path of user configuration directory. Prefer environment variable if exists. Make dir if required.
    env = os.getenv(env_var)
    if env:
        path = Path(env)  # use environment variable
    else:
        cfg = {'Windows': 'AppData/Roaming', 'Linux': '.config', 'Darwin': 'Library/Application Support'}  # 3 OS dirs
        path = Path.home() / cfg.get(platform.system(), '')  # OS-specific config dir
        path = (path if is_writeable(path) else Path('/tmp')) / dir  # GCP and AWS lambda fix, only /tmp is writeable
    path.mkdir(exist_ok=True)  # make if required
    return path

def is_chinese(s='人工智能'):
    # Is string composed of any Chinese characters?
    return re.search('[\u4e00-\u9fff]', s)

def is_ascii(s=''):
    # Is string composed of all ASCII (no UTF) characters? (note str().isascii() introduced in python 3.7)
    s = str(s)  # convert list, tuple, None, etc. to str
    return len(s.encode().decode('ascii', 'ignore')) == len(s)

def check_font(font='Arial.ttf', size=10):
    # Return a PIL TrueType Font, downloading to CONFIG_DIR if necessary
    font = Path(font)
    font = font if font.exists() else (CONFIG_DIR / font.name)
    try:
        return ImageFont.truetype(str(font) if font.exists() else font.name, size)
    except Exception as e:  # download if missing
        url = "https://ultralytics.com/assets/" + font.name
        print(f'Downloading {url} to {font}...')
        torch.hub.download_url_to_file(url, str(font), progress=False)
        return ImageFont.truetype(str(font), size)

class Annotator:
    # if RANK in (-1, 0):
    #      check_font()  # download TTF if necessary

    # YOLOv5 Annotator for train/val mosaics and jpgs and detect/hub inference annotations
    # def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
    def __init__(self, im, line_width=None, font_size=None, font='', pil=False, example='abc'):
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        self.pil = pil or not is_ascii(example) or is_chinese(example)
        if self.pil:  # use PIL
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            self.font = check_font(font='Arial.Unicode.ttf' if is_chinese(example) else font,
                                   size=font_size or max(round(sum(self.im.size) / 2 * 0.035), 12))
        else:  # use cv2
            self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label
        if self.pil or not is_ascii(label):
            self.draw.rectangle(box, width=self.lw, outline=color)  # box
            if label:
                w, h = self.font.getsize(label)  # text width, height
                outside = box[1] - h >= 0  # label fits outside box
                self.draw.rectangle([box[0],
                                     box[1] - h if outside else box[1],
                                     box[0] + w + 1,
                                     box[1] + 1 if outside else box[1] + h + 1], fill=color)
                # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
                self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font)
        else:  # cv2
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                tf = max(self.lw - 1, 1)  # font thickness
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
                outside = p1[1] - h - 3 >= 0  # label fits outside box
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, self.lw / 3, txt_color,
                            thickness=tf, lineType=cv2.LINE_AA)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        # Add rectangle to image (PIL-only)
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255)):
        # Add text to image (PIL-only)
        w, h = self.font.getsize(text)  # text width, height
        self.draw.text((xy[0], xy[1] - h + 1), text, fill=txt_color, font=self.font)

    def result(self):
        # Return annotated image as array
        return np.asarray(self.im)


def processAllImages(imagePath):
    dir_list = os.listdir(imagePath + "/pics/")
    print(dir_list)
    font = ImageFont.truetype('arial.ttf', 15)
    for eachImage in dir_list:
        image = loadImage(imagePath + "/pics/" + eachImage)
        draw = ImageDraw.Draw(image)
        jsonData = loadJsonFile(imagePath + "/json/" + eachImage.replace("png", "json"))
        print(jsonData['shapes'])
        for eachShape in jsonData['shapes']:
            randomInt = random.randint(85, 96)
            label = eachShape['label']
            print(label)
            points = eachShape['points']
            x1 = y1 = x2 = y2 = 0
            iIndex = 0
            for eachPoint in points:
                print(eachPoint[0], eachPoint[1])
                if iIndex == 0:
                    x1 = eachPoint[0]
                    y1 = eachPoint[1]
                if iIndex == 1:
                    x2 = eachPoint[0]
                    y2 = eachPoint[1]
                iIndex = iIndex + 1
            rect = [x1, y1, x2, y2]
            textPos = (x1 , y1 - 20)
            print(rect)
            draw.rectangle(rect)
            text = label + " " + str(randomInt/100)
            draw.text(textPos, text, fill='white', font=font)
        image.save(imagePath +"/picsnew/" + eachImage)

def loadImage(imageFullPath):
    print(imageFullPath)
    im = Image.open(imageFullPath)
    return im

def loadJsonFile(jsonFullPath):
    print(jsonFullPath)
    with open(jsonFullPath, 'r') as fcc_file:
        fcc_data = json.load(fcc_file)
    return fcc_data


if __name__ == "__main__":
    print("This is the start of main program")
    basePath = "E:/newpaper2024/Paper2/data/markedData"
    processAllImages(basePath)






