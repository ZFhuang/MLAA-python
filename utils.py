import os
import shutil
import cv2
import numpy as np


def load_img_float(path):
    """
    return [0, 1]
    """
    img = cv2.imread(path, -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img/255


def save_img_float(path, img):
    """
    input [0, 1]
    """
    img = (img*255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def save_gray_img_float(path, img_gray):
    """
    input [0, 1]
    """
    img = (img_gray*255).astype(np.uint8)
    img = np.dstack((img, img, img))
    cv2.imwrite(path, img)


def check_folder(dir):
    """rasie FileNotFoundError"""
    if not os.path.isdir(dir):
        raise FileNotFoundError(dir+' is not exist!')


def check_file(file):
    """rasie FileNotFoundError"""
    if not os.path.exists(file):
        raise FileNotFoundError(file+' is not exist!')


def init_folder(dir):
    if not os.path.isdir(dir):
        try:
            os.mkdir(dir)
            print('Folder init: '+dir)
        except FileNotFoundError:
            os.makedirs(dir)
            print('Folders init: '+dir)


def refresh_folder(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir)
    init_folder(dir)


def filter_files_with_suffix(list_files, str_suffix):
    """
    suffix must be the last 4 latter
    """
    list_ret = []
    for file in list_files:
        if file[-4:] == str_suffix:
            list_ret.append(file)
    return list_ret
