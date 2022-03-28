#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Author : Gabriel
@About :
'''
import tensorflow as tf
from tensorflow.keras import backend as K

from utils.utils_bbox import get_anchors_and_decode


def box_iou(box1, box2):
    '''
    计算预测框和真实框的IOU
    '''
    '''
    预测框
    计算左上角和右下角的坐标
    '''
    box1 = K.expand_dims(box1, -2)
    box1_xy = box1[..., :2]
    box1_wh = box1[..., 2:4]
    box1_wh_half = box1_wh / 2.0
    box1_min = box1_xy - box1_wh_half
    box1_max = box1_xy + box1_wh_half

    '''
    真实框
    计算左上角和右下角的坐标
    '''
    box2 = K.expand_dims(box2, 0)
    box2_xy = box2[..., :2]
    box2_wh = box2[..., 2:4]
    box2_wh_half = box2_wh / 2.0
    box2_min = box2_xy - box2_wh_half
    box2_max = box2_xy + box2_wh_half

    '''
    计算IOU
    '''
    intersect_min = K.maximum(box1_min, box2_min)
    intersect_max = K.minimum(box1_max, box2_max)
    intersect_wh = K.maximum(intersect_max - intersect_min, 0.0)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    box1_area = box1_wh[..., 0] * box1_wh[..., 1]
    box2_area = box2_wh[..., 0] * box2_wh[..., 1]
    iou = intersect_area / (box1_area + box2_area - intersect_area)

    return iou


def yolo_loss(args, input_shape, anchors, anchors_mask, num_classes, ignore_th=0.5):
    '''
    计算损失
    '''
    num_layers = len(anchors_mask)
    '''
    将预测结果和真实框分开，args是[*model_body.output, *y_true]
    y_true是一个列表，包含三个特征层，shape分别为:
    (m,13,13,3,85)
    (m,26,26,3,85)
    (m,52,52,3,85)
    yolo_outputs是一个列表，包含三个特征层，shape分别为:
    (m,13,13,3,85)
    (m,26,26,3,85)
    (m,52,52,3,85)
    '''
    y_true = args[num_layers:]
    y_pred = args[:num_layers]

    # [416,416]
    input_shape = K.cast(input_shape, K.dtype(y_true[0]))
    # [13,13],[26,26],[52,52]
    grid_shapes = [K.cast(K.shape(y_pred[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]

    '''
    取出图片
    m = batch_size
    '''
    bs = K.shape(y_pred[0])[0]

    loss = 0
    num_pos = 0

    '''
    以第一个特征层[m,13,13,3,85]为例
    '''
    for l in range(num_layers):
        '''
        取出该特征层中存在目标的点的位置
        [m,13,13,3,1]
        '''
        obj_mask = y_true[l][..., 4:5]

        '''
        取出其对应的种类
        [m,13,13,3,80]
        '''
        true_class_probs = y_true[l][..., 5:]

        '''
        对模型输出的特征层进行处理，获得四个值:
        grid        (13,13,1,2)     网格坐标
        raw_pred    (m,13,13,3,85)  尚未处理的预测结果
        pred_xy     (m,13,13,3,2)   解码后的中心坐标
        pred_wh     (m,13,13,3,2)   解码后的宽高坐标
        '''
        grid, raw_pred, pred_xy, pred_wh = get_anchors_and_decode(y_pred[l], anchors[anchors_mask[l]],
                                                                  num_classes, input_shape, calc_loss=True)

        '''
        解码后的预测的box的位置
        [m,13,13,3,4]
        '''
        pred_box = K.concatenate([pred_xy, pred_wh])

        '''
        找到负样本，创建数组
        '''
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        obj_mask_bool = K.cast(obj_mask, 'bool')

        '''
        对每张图片计算ignore_mask
        '''

        def loop_body(b, ignore_mask):
            '''
            取出n个真实框
            [n,4]
            '''
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], obj_mask_bool[b, ..., 0])

            '''
            计算IOU
            '''
            iou = box_iou(pred_box[b], true_box)

            '''
            获得最大重合度
            '''
            best_iou = K.max(iou, axis=-1)

            '''
            若预测框和真实框最大IOU小于ignore_th，则认为预测框没有与之对应的真实框
            '''
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_th, K.dtype(true_box)))

            return b + 1, ignore_mask

        _, ignore_mask = tf.while_loop(lambda b, *args: b < bs, loop_body, [0, ignore_mask])

        '''
        用ignore_mask提取出作为负样本的特征点
        [m,13,13,3] -> [m,13,13,3,1]
        '''
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)
        '''
        将真实框进行编码，与预测框保持相同格式
        '''
        raw_true_xy = y_true[l][..., :2] * grid_shapes[l][:] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchors_mask[l]] * input_shape[::-1])

        '''
        若obj_mask真实存在目标则保持wh
        '''
        raw_true_wh = K.switch(obj_mask, raw_true_wh, K.zeros_like(raw_true_wh))

        '''
        y_true[...,2:3]和y_true[...,3:4]
        表示真实框的宽高，二者均在0-1之间
        真实框越大，比重越小，小框的比重更大。
        '''
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        '''
        计算中心点偏移情况
        '''
        xy_loss = obj_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2], from_logits=True)

        '''
        计算高宽损失
        '''
        wh_loss = obj_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])

        '''
        如果该位置本来有框，那么计算1与置信度的交叉熵
        如果该位置本来没有框，那么计算0与置信度的交叉
        '''
        conf_loss = obj_mask * K.binary_crossentropy(obj_mask, raw_pred[..., 4:5], from_logits=True) + \
                    (1 - obj_mask) * K.binary_crossentropy(obj_mask, raw_pred[..., 4:5], from_logits=True) * ignore_mask
        class_loss = obj_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

        '''
        合并所有损失
        '''
        xy_loss = K.sum(xy_loss)
        wh_loss = K.sum(wh_loss)
        conf_loss = K.sum(conf_loss)
        class_loss = K.sum(class_loss)
        '''
        计算正样本数量
        '''
        num_pos += tf.maximum(K.sum(K.cast(obj_mask, tf.float32)), 1)
        loss += xy_loss + wh_loss + conf_loss + class_loss

    loss = loss / num_pos

    return loss
