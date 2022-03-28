#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Author : Gabriel
@About : 
'''
from tensorflow import keras
from tensorflow.keras import layers, models

from nets.darknet import darknet_body
from nets.loss import yolo_loss

padding_type = lambda x: 'valid' if x == (2, 2) else 'same'


def five_conv(x, num_filters):
    x = layers.Conv2D(num_filters, kernel_size=(1, 1), kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                      kernel_regularizer=keras.regularizers.l2(5e-4), strides=(1, 1), padding=padding_type((1, 1)),
                      use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(num_filters * 2, kernel_size=(3, 3),
                      kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                      kernel_regularizer=keras.regularizers.l2(5e-4), strides=(1, 1), padding=padding_type((1, 1)),
                      use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(num_filters, kernel_size=(1, 1), kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                      kernel_regularizer=keras.regularizers.l2(5e-4), strides=(1, 1), padding=padding_type((1, 1)),
                      use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(num_filters * 2, kernel_size=(3, 3),
                      kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                      kernel_regularizer=keras.regularizers.l2(5e-4), strides=(1, 1), padding=padding_type((1, 1)),
                      use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(num_filters, kernel_size=(1, 1), kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                      kernel_regularizer=keras.regularizers.l2(5e-4), strides=(1, 1), padding=padding_type((1, 1)),
                      use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    return x


def yolo_head(x, num_filters, out_filters):
    x = layers.Conv2D(num_filters * 2, kernel_size=(3, 3),
                      kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                      kernel_regularizer=keras.regularizers.l2(5e-4), strides=(1, 1), padding=padding_type((1, 1)),
                      use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    y = layers.Conv2D(out_filters, kernel_size=(1, 1),
                      kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                      kernel_regularizer=keras.regularizers.l2(5e-4), strides=(1, 1), padding=padding_type((1, 1)))(x)

    return y


def yolo_body(input_shape, anchors_mask, num_classes):
    inputs = layers.Input(input_shape)
    '''
    获得三个有效特征层
    feat1: [52,52,256]
    feat2: [26,26,512]
    feat3: [13,13,1024]
    '''
    feat1, feat2, feat3 = darknet_body(inputs)

    '''
    第一个特征层
    [bs,13,13,3,85]
    '''
    # [13,13,1024] -> [13,13,512] -> [13,13,1024] -> [13,13,512] -> [13,13,1024] -> [13,13,512]
    x = five_conv(feat3, 512)
    y1 = yolo_head(x, 512, len(anchors_mask[0]) * (num_classes + 5))

    # [13,13,512] -> [13,13,256] -> [26,26,256]
    x = layers.Conv2D(256, kernel_size=(1, 1), kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                      kernel_regularizer=keras.regularizers.l2(5e-4), strides=(1, 1), padding=padding_type((1, 1)),
                      use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.UpSampling2D(2)(x)

    # [26,26,256] + [26,26,512] -> [26,26,768]
    x = layers.Concatenate()([x, feat2])

    '''
    第二个特征层
    [bs,26,26,3,85]
    '''
    # [26,26,768] -> [26,26,256] -> [26,26,512] -> [26,26,256] -> [26,26,512] -> [26,26,256]
    x = five_conv(x, 256)
    y2 = yolo_head(x, 256, len(anchors_mask[1]) * (num_classes + 5))

    # [26,26,256] -> [26,26,128] -> [52,52,128]
    x = layers.Conv2D(128, kernel_size=(1, 1), kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                      kernel_regularizer=keras.regularizers.l2(5e-4), strides=(1, 1), padding=padding_type((1, 1)),
                      use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.UpSampling2D(2)(x)

    # [52,52,128] + [52,52,256] -> [26,26,384]
    x = layers.Concatenate()([x, feat1])

    '''
    第三个特征层
    [bs,52,52,3,85]
    '''
    # [52,52,384] -> [52,52,128] -> [52,52,256] -> [52,52,128] -> [52,52,256] -> [52,52,128]
    x = five_conv(x, 128)
    y3 = yolo_head(x, 128, len(anchors_mask[2]) * (num_classes + 5))

    return models.Model(inputs, [y1, y2, y3])


def get_train_model(model_body, input_shape, num_classes, anchors, anchors_mask):
    y_true = [layers.Input(shape=(input_shape[0] // {0: 32, 1: 16, 2: 8}[l], input_shape[1] // {0: 32, 1: 16, 2: 8}[l], \
                                  len(anchors_mask[l]), num_classes + 5)) for l in range(len(anchors_mask))]
    model_loss = layers.Lambda(
        yolo_loss,
        output_shape=(1,),
        name='yolo_loss',
        arguments={'input_shape': input_shape, 'anchors': anchors, 'anchors_mask': anchors_mask,
                   'num_classes': num_classes}
    )([*model_body.output, *y_true])
    model = models.Model([model_body.input, *y_true], model_loss)
    return model
