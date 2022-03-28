#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Author : Gabriel
@About : 
'''
import tensorflow as tf
from tensorflow.keras import backend as K


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_img):
    '''
    调整box，使其与真实图片匹配
    '''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    if letterbox_img:
        '''
        offset:     图片有效区域相对于图片左上角的偏移情况
        new_shape:  宽高缩放情况
        '''
        new_shape = K.round(image_shape * K.min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2.0 / input_shape
        scale = input_shape / new_shape

        box_yx = (box_yx - offset) * scale
        box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxs = box_yx + (box_hw / 2.)
    boxes = K.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxs[..., 0:1], box_maxs[..., 1:2]])
    boxes *= K.concatenate([image_shape, input_shape])

    return boxes


def get_anchors_and_decode(feats, anchors, num_classes, input_shape, calc_loss=False):
    '''
    将每个特征层的输出转换为真实值
    '''
    num_anchors = len(anchors)
    # 特征层的高宽
    grid_shape = K.shape(feats)[1:3]
    '''
    获得各个特征点的坐标信息
    [13,13,num_anchors,2]
    '''
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, num_anchors, 1])
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], num_anchors, 1])
    grid = K.cast(K.concatenate([grid_x, grid_y]), K.dtype(feats))

    '''
    展开先验框
    [13,13,num_anchors,2]
    '''
    anchors_ = K.reshape(K.constant(anchors), [1, 1, num_anchors, 2])
    anchors_ = K.tile(anchors_, [grid_shape[0], grid_shape[1], 1, 1])

    '''
    修改预测结果的shape
    [batch_size,13,13,3,85]
    85:
    4(中心宽高的调整参数)+1(框的置信度)+80(分类的置信度)
    '''
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    '''
    解码先验框并进行归一化
    '''
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_ / K.cast(input_shape[::-1], K.dtype(feats))

    '''
    获得预测框的置信度
    '''
    box_conf = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    '''
    训练时返回:
        grid,feats,box_xy,box_wh
    预测时返回:
        box_xy,box_wh,box_conf,box_class_probs
    '''
    if calc_loss:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_conf, box_class_probs


def decode_box(outputs, anchors, num_classes, input_shape, anchor_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
               max_boxes=100, threshold=0.5, nms_iou=0.5, letterbox_img=True):
    '''
    解码先验框
    '''
    image_shape = K.reshape(outputs[-1], [-1])

    box_xy = []
    box_wh = []
    box_conf = []
    box_class_probs = []
    for i in range(len(anchor_mask)):
        sub_box_xy, sub_box_wh, sub_box_conf, sub_box_class_probs = get_anchors_and_decode(
            outputs[i], anchors[anchor_mask[i]], num_classes, input_shape
        )
        box_xy.append(K.reshape(sub_box_xy, [-1, 2]))
        box_wh.append(K.reshape(sub_box_wh, [-1, 2]))
        box_conf.append(K.reshape(sub_box_conf, [-1, 1]))
        box_class_probs.append(K.reshape(sub_box_class_probs, [-1, num_classes]))

    box_xy = K.concatenate(box_xy, axis=0)
    box_wh = K.concatenate(box_wh, axis=0)
    box_conf = K.concatenate(box_conf, axis=0)
    box_class_probs = K.concatenate(box_class_probs, axis=0)

    '''
    去掉填充时的灰度条
    '''
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_img)
    box_scores = box_conf * box_class_probs

    '''
    判断是否大于阈值
    '''
    mask = box_scores >= threshold
    max_boxes_ = K.constant(max_boxes, dtype='int32')
    boxes_out = []
    scores_out = []
    classes_out = []
    for i in range(num_classes):
        '''
        取出大于阈值的框和得分
        '''
        class_boxes = tf.boolean_mask(boxes, mask[:,i])
        class_box_scores = tf.boolean_mask(box_scores[:, i], mask[:, i])

        '''
        NMS
        '''
        nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_, iou_threshold=nms_iou)

        '''
        获得NMS的结果：位置、得分、分类
        '''
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * i

        boxes_out.append(class_boxes)
        scores_out.append(class_box_scores)
        classes_out.append(classes)

    boxes_out = K.concatenate(boxes_out, axis=0)
    scores_out = K.concatenate(scores_out, axis=0)
    classes_out = K.concatenate(classes_out, axis=0)

    return boxes_out, scores_out, classes_out
