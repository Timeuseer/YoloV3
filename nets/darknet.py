#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Author : Gabriel
@About : backbone
'''
from tensorflow import keras
from tensorflow.keras import layers


def darknet_body(x):
    padding_type = lambda x: 'valid' if x == (2, 2) else 'same'
    '''
    Block 1
    [416,416,3] -> [416,416,32]
    '''
    x = layers.Conv2D(32, kernel_size=(3, 3), kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                      kernel_regularizer=keras.regularizers.l2(5e-4), strides=(1, 1), padding=padding_type((1, 1)),
                      use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    '''
    Block2
    [416,416,32] -> [208,208,64]
    '''
    x = layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                      kernel_regularizer=keras.regularizers.l2(5e-4), strides=(2, 2), padding=padding_type((2, 2)),
                      use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    for _ in range(1):
        y = layers.Conv2D(64 // 2, kernel_size=(1, 1), kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                          kernel_regularizer=keras.regularizers.l2(5e-4), strides=(1, 1), padding=padding_type((1, 1)),
                          use_bias=False)(x)
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU(alpha=0.1)(y)

        y = layers.Conv2D(64, kernel_size=(3, 3), kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                          kernel_regularizer=keras.regularizers.l2(5e-4), strides=(1, 1), padding=padding_type((1, 1)),
                          use_bias=False)(y)
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU(alpha=0.1)(y)

        x = layers.Add()([x, y])

    '''
    Block 3
    [208,208,64] -> [104,104,128]
    '''
    x = layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = layers.Conv2D(128, kernel_size=(3, 3), kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                      kernel_regularizer=keras.regularizers.l2(5e-4), strides=(2, 2), padding=padding_type((2, 2)),
                      use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    for _ in range(2):
        y = layers.Conv2D(128 // 2, kernel_size=(1, 1), kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                          kernel_regularizer=keras.regularizers.l2(5e-4), strides=(1, 1), padding=padding_type((1, 1)),
                          use_bias=False)(x)
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU(alpha=0.1)(y)

        y = layers.Conv2D(128, kernel_size=(3, 3), kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                          kernel_regularizer=keras.regularizers.l2(5e-4), strides=(1, 1), padding=padding_type((1, 1)),
                          use_bias=False)(y)
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU(alpha=0.1)(y)

        x = layers.Add()([x, y])

    '''
    Block 4
    [104,104,128] -> [52,52,256]
    '''
    x = layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = layers.Conv2D(256, kernel_size=(3, 3), kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                      kernel_regularizer=keras.regularizers.l2(5e-4), strides=(2, 2), padding=padding_type((2, 2)),
                      use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    for _ in range(8):
        y = layers.Conv2D(256 // 2, kernel_size=(1, 1), kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                          kernel_regularizer=keras.regularizers.l2(5e-4), strides=(1, 1), padding=padding_type((1, 1)),
                          use_bias=False)(x)
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU(alpha=0.1)(y)

        y = layers.Conv2D(256, kernel_size=(3, 3), kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                          kernel_regularizer=keras.regularizers.l2(5e-4), strides=(1, 1), padding=padding_type((1, 1)),
                          use_bias=False)(y)
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU(alpha=0.1)(y)

        x = layers.Add()([x, y])

    feat1 = x

    '''
    Block 5
    [52,52,256] -> [26,26,512]
    '''
    x = layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = layers.Conv2D(512, kernel_size=(3, 3), kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                      kernel_regularizer=keras.regularizers.l2(5e-4), strides=(2, 2), padding=padding_type((2, 2)),
                      use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    for _ in range(8):
        y = layers.Conv2D(512 // 2, kernel_size=(1, 1), kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                          kernel_regularizer=keras.regularizers.l2(5e-4), strides=(1, 1), padding=padding_type((1, 1)),
                          use_bias=False)(x)
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU(alpha=0.1)(y)

        y = layers.Conv2D(512, kernel_size=(3, 3), kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                          kernel_regularizer=keras.regularizers.l2(5e-4), strides=(1, 1), padding=padding_type((1, 1)),
                          use_bias=False)(y)
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU(alpha=0.1)(y)

        x = layers.Add()([x, y])

    feat2 = x

    '''
    Block 6
    [26,26,512] -> [13,13,1024]
    '''
    x = layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = layers.Conv2D(1024, kernel_size=(3, 3), kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                      kernel_regularizer=keras.regularizers.l2(5e-4), strides=(2, 2), padding=padding_type((2, 2)),
                      use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    for _ in range(4):
        y = layers.Conv2D(1024 // 2, kernel_size=(1, 1),
                          kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                          kernel_regularizer=keras.regularizers.l2(5e-4), strides=(1, 1), padding=padding_type((1, 1)),
                          use_bias=False)(x)
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU(alpha=0.1)(y)

        y = layers.Conv2D(1024, kernel_size=(3, 3), kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                          kernel_regularizer=keras.regularizers.l2(5e-4), strides=(1, 1), padding=padding_type((1, 1)),
                          use_bias=False)(y)
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU(alpha=0.1)(y)

        x = layers.Add()([x, y])

    feat3 = x

    return feat1, feat2, feat3
