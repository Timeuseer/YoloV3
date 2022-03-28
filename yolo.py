#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Author : Gabriel
@About : 
'''
import colorsys
import os
import time
import yaml
import numpy as np
import tensorflow as tf
from PIL import ImageFont, ImageDraw
from tensorflow.keras import layers, models

from nets.yolo import yolo_body
from utils.utils import cvtColor, get_anchors, get_classes, preprocess_input, resize_img
from utils.utils_bbox import decode_box


class YOLO(object):
    _defaults = yaml.load(open('config/yolo_config.yaml', 'r', encoding='utf-8'))

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return f"Unrecognized attribute name ' {n} '"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for nama, value in kwargs.items():
            setattr(self, nama, value)

        '''
        获得种类和先验框
        '''
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors, self.num_anchors = get_anchors(self.anchors_path)

        '''
        画框的颜色
        '''
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.generate()

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        self.yolo_model = yolo_body([None, None, 3], self.anchors_mask, self.num_classes)
        self.yolo_model.load_weights(self.model_path, by_name=True)

        print('{} model, anchors, and classes loaded.'.format(model_path))
        self.input_img_shape = layers.Input([2, ], batch_size=1)
        inputs = [*self.yolo_model.output, self.input_img_shape]
        outputs = layers.Lambda(decode_box, output_shape=(1,), name='yolo_eval',
                                arguments={
                                    'anchors': self.anchors,
                                    'num_classes': self.num_classes,
                                    'input_shape': self.input_shape,
                                    'anchor_mask': self.anchors_mask,
                                    'threshold': self.confidence,
                                    'nms_iou': self.nms_iou,
                                    'max_boxes': self.max_boxes,
                                    'letterbox_img': self.letterbox_img
                                })(inputs)
        self.yolo_model = models.Model([self.yolo_model.input, self.input_img_shape], outputs)

    @tf.function
    def get_pred(self, img_data, input_img_shape):
        out_boxes, out_scores, out_classes = self.yolo_model([img_data, input_img_shape], training=False)
        return out_boxes, out_scores, out_classes

    def detect_img(self, img):
        '''
        转换为RGB图像
        '''
        img = cvtColor(img)

        '''
        添加灰度条
        '''
        img_data = resize_img(img, (self.input_shape[1], self.input_shape[0]), self.letterbox_img)
        img_data = np.expand_dims(preprocess_input(np.array(img_data, dtype=np.float32)), 0)

        '''
        进行预测
        '''
        input_img_shape = np.expand_dims(np.array([img.size[1], img.size[0]], dtype=np.float32), 0)
        out_boxes, out_scores, out_classes = self.get_pred(img_data, input_img_shape)

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        '''
        设置字体和边框
        '''
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * img.size[1] + 0.5).astype('int32'))
        thickness = int(max((img.size[0] + img.size[1]) // np.mean(self.input_shape), 1))

        '''
        绘制图像
        '''
        for i, c in list(enumerate(out_classes)):
            pred_class = self.class_names[int(c)]
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(img.size[1], np.floor(bottom).astype('int32'))
            right = min(img.size[0], np.floor(right).astype('int32'))

            label = f'{pred_class} {score:.2f}'
            draw = ImageDraw.Draw(img)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return img

    def get_FPS(self, img, test_interval):
        '''
        转换为RGB图像
        '''
        img = cvtColor(img)

        '''
        添加灰度条
        '''
        img_data = resize_img(img, (self.input_shape[1], self.input_shape[0]), self.letterbox_img)
        img_data = np.expand_dims(preprocess_input(np.array(img_data, dtype=np.float32)), 0)

        '''
        进行预测
        '''
        input_img_shape = np.expand_dims(np.array([img.size[1], img.size[0]], dtype=np.float32), 0)
        out_boxes, out_scores, out_classes = self.get_pred(img_data, input_img_shape)

        t1 = time.time()
        for _ in range(test_interval):
            out_boxes, out_scores, out_classes = self.get_pred(img_data, input_img_shape)
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_map_txt(self, img_id, img, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + img_id + ".txt"), "w")
        img_shape = np.array(np.shape(img)[0:2])
        '''
        将图像转化为RGB模式
        '''
        img = cvtColor(img)

        '''
        根据需求，觉得是否给图像进行不失真的resize
        '''
        img_data = resize_img(img, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        '''
        增加batch_size维度，并进行图像预处理，归一化
        '''
        img_data = preprocess_input(np.expand_dims(np.array(img_data, dtype=np.float32), 0))

        preds = self.get_pred(img_data).numpy()
        '''
        将预测结果进行解码
        '''
        results = self.bbox_util.decode_box(preds, self.anchors, img_shape, self.input_shape,
                                            self.letterbox_image, confidence=self.confidence)
        '''
        如果没有检测到物体，则返回
        '''
        if len(results[0]) <= 0:
            return

        top_label = results[0][:, 4]
        top_conf = results[0][:, 5]
        top_boxes = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            top, left, bottom, right = box

            if predicted_class not in class_names:
                continue

            f.write(f"{predicted_class} {score[:6]} {str(int(left))} {int(top)} {int(right)} {int(bottom)} \n")

        f.close()
        return
