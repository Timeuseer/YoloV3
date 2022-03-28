#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Author : Gabriel
@About : 
'''
import numpy as np
from PIL import Image
from functools import reduce


def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def cvtColor(img):
    '''
    将图像转换成RGB,放在灰度图在预测时报错
    '''
    if len(np.shape(img)) == 3 and np.shape(img)[-2] == 3:
        return img
    else:
        img = img.convert('RGB')
        return img


def resize_img(img, size, letterbox_img):
    '''
    对图像进行尺寸变换
    '''
    iw, ih = img.size
    w, h = size
    if letterbox_img:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        img = img.resize((nw, nh), Image.BICUBIC)
        new_img = Image.new('RGB', size, (128, 128, 128))
        new_img.paste(img, ((w - nw) // 2, (h - nh) // 2))
    else:
        new_img = img.resize((w, h), Image.BICUBIC)

    return new_img


def get_classes(classes_path):
    '''
    获得类
    '''
    with open(classes_path, encoding='utf-8') as f:
        classes_names = f.readlines()
    classes_names = [c.strip() for c in classes_names]

    return classes_names, len(classes_names)


def get_anchors(anchors_path):
    '''
    获得先验框
    '''
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)


def preprocess_input(image):
    '''
    归一化
    '''
    image /= 255.0
    return image
