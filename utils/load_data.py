#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Author : Gabriel
@About :
'''
import math
import cv2
import numpy as np
from PIL import Image
from tensorflow import keras
from utils.utils import cvtColor, preprocess_input
from random import shuffle


class YoloDatasets(keras.utils.Sequence):
    def __init__(self, annotation_lines, input_shape, anchors, batch_size, num_classes, anchors_mask, train):
        self.annotation_lines = annotation_lines
        self.length = len(self.annotation_lines)
        self.input_shape = input_shape
        self.anchors = anchors
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.anchors_mask = anchors_mask
        self.train = train

    def __len__(self):
        return math.ceil(len(self.annotation_lines) / float(self.batch_size))

    def __getitem__(self, index):
        img_data = []
        box_data = []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):
            i = i % self.length
            img, box = self.get_random_data(self.annotation_lines[i], self.input_shape, random=self.train)
            img_data.append(preprocess_input(np.array(img)))
            box_data.append(box)

        img_data = np.array(img_data)
        box_data = np.array(box_data)
        y_true = self.preprocess_true_boxes(box_data, self.input_shape, self.anchors, self.num_classes)

        return [img_data, *y_true], np.zeros(self.batch_size)

    def generate(self):
        i = 0
        while True:
            img_data = []
            box_data = []
            for b in range(self.batch_size):
                if i == 0:
                    np.random.shuffle(self.annotation_lines)
                img, box = self.get_random_data(self.annotation_lines[i], self.input_shape, random=self.train)
                i = (i + 1) % self.length
                img_data.append(preprocess_input(np.array(img)))
                box_data.append(box)

            img_data = np.array(img_data)
            box_data = np.array(box_data)
            y_true = self.preprocess_true_boxes(box_data, self.input_shape, self.anchors, self.num_classes)

            yield img_data, y_true[0], y_true[1], y_true[2]

    def on_epoch_begin(self):
        shuffle(self.annotation_lines)

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, max_boxes=100, jitter=0.3, hue=0.1, sat=1.5, val=1.5,
                        random=True):
        line = annotation_line.split()

        '''
        读取图片并转换成RGB
        '''
        img = Image.open(line[0])
        img = cvtColor(img)

        '''
        获得图像的宽高和目标宽高
        '''
        iw, ih = img.size
        h, w = input_shape

        '''
        获得预测框
        '''
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            '''
            进行灰度条填充
            '''
            img = img.resize((nw, nh), Image.BICUBIC)
            new_img = Image.new('RGB', (w, h), (128, 128, 128))
            new_img.paste(img, (dx, dy))
            img_data = np.array(new_img, np.float32)

            '''
            对真实框进行调整
            '''
            box_data = np.zeros((max_boxes, 5))
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                if len(box) > max_boxes:
                    box = box[:max_boxes]
                box_data[:len(box)] = box

            return img_data, box_data

        '''
        对图像进行缩放并进行长和宽的扭曲
        '''
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw * new_ar)
        img = img.resize((nw, nh), Image.BICUBIC)

        '''
        将图像多图的部分加上灰条
        '''
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_img = Image.new('RGB', (w, h), (128, 128, 128))
        new_img.paste(img, (dx, dy))
        img = new_img

        '''
        翻转图像
        '''
        flip = self.rand() < 0.5
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        '''
        色域扭曲
        '''
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < 0.5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < 0.5 else 1 / self.rand(1, val)
        x = cv2.cvtColor(np.array(img, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 0] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        img_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

        '''
        对真实框进行调整
        '''
        box_data = np.zeros((max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            bow_w = box[:, 2] - box[:, 0]
            bow_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(bow_w > 1, bow_h > 1)]
            if len(box) > max_boxes:
                box = box[:max_boxes]
            box_data[:len(box)] = box

        return img_data, box_data

    def preprocess_true_boxes(self, true_boxes, input_shape, anchors, num_classes):
        assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'

        '''
        获得框的坐标和图片大小
        '''
        true_boxes = np.array(true_boxes, dtype=np.float32)
        input_shape = np.array(input_shape, dtype=np.int32)

        num_layers = len(self.anchors_mask)
        bs = true_boxes.shape[0]
        grid_shape = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]

        '''
        [bs,13,13,3,85]
        [bs,26,26,3,85]
        [bs,52,52,3,85]
        '''
        y_true = [np.zeros((bs, grid_shape[l][0], grid_shape[l][1], len(self.anchors_mask[l]), 5 + num_classes),
                           dtype=np.float32) for l in range(num_layers)]

        '''
        通过计算获得真实框的中心和宽高
        中心点:    [bs,n,2]
        宽高：     [bs,n,2]
        '''
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]

        '''
        归一化
        '''
        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
        true_boxes[..., 2:4] = boxes_xy / input_shape[::-1]

        anchors = np.expand_dims(anchors, axis=0)
        anchors_maxes = anchors / 2.0
        anchors_mines = - anchors_maxes

        '''
        高宽要大于0才有效
        '''
        valid_mask = boxes_wh[..., 0] > 0
        for b in range(bs):
            wh = boxes_wh[b, valid_mask[b]]
            if len(wh) == 0:
                continue
            wh = np.expand_dims(wh, axis=-2)
            box_maxes = wh / 2.0
            box_mines = - box_maxes

            '''
            计算所有真实框和先验框的IOU
            intersect_area  [n,9]
            box_area        [n,1]
            anchor_area     [1,9]
            iou             [n,9]
            '''
            intersect_mines = np.maximum(box_mines, anchors_mines)
            intersect_maxes = np.minimum(box_maxes, anchors_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mines, 0.0)
            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

            box_area = wh[..., 0] * wh[..., 1]
            anchor_area = anchors[..., 0] * wh[..., 1]

            iou = intersect_area / (box_area + anchor_area - intersect_area)

            best_anchor = np.argmax(iou, axis=-1)
            for t, n in enumerate(best_anchor):
                '''
                找到每个真实框所属的特征层
                '''
                for l in range(num_layers):
                    if n in self.anchors_mask[l]:
                        '''
                        向下取整，找到真实框所属的特征层对应的x,y轴坐标
                        '''
                        i = np.floor(true_boxes[b, t, 0] * grid_shape[l][1]).astype('int32')
                        j = np.floor(true_boxes[b, t, 1] * grid_shape[l][0]).astype('int32')

                        # 当前这个特征点的第k个先验框
                        k = self.anchors_mask[l].index(n)

                        # 当前这个真实框的种类
                        c = true_boxes[b, t, 4].astype('int32')

                        '''
                        [bs,13,13,3,85]
                        [bs,26,26,3,85]
                        [bs,52,52,3,85]
                        85 = 4 + 1 + 80
                        '''
                        y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                        y_true[l][b, j, i, k, 4] = 1
                        y_true[l][b, j, i, k, 5 + c] = 1

        return y_true
