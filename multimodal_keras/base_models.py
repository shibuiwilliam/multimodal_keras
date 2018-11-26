import os
import random
import numpy as np
import pandas as pd

import keras
from keras import backend as K
from keras import (Model, Input, optimizers, losses, callbacks)
from keras.layers import (Activation, Dropout, AlphaDropout, Conv1D, Conv2D,
                          Reshape, Lambda, GlobalMaxPooling1D, MaxPool2D,
                          GlobalAveragePooling2D, Dense, MaxPool1D,
                          GlobalMaxPooling2D, BatchNormalization, Embedding,
                          Concatenate, Maximum, Add)


def numerical_mlp(inputs, output_size):
    x = Dense(512, activation="relu")(inputs)
    x = Dropout(0.2)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(output_size)(x)
    return x


def text_character_level_cnn(inputs, output_size):
    x = Embedding(0xffff, 256)(inputs)
    convs = []
    for i, p in enumerate([2, 3, 4, 5]):
        _x = Conv1D(
            filters=256, kernel_size=p, strides=(p // 2), padding="same")(x)
        _x = Activation("tanh")(_x)
        _x = GlobalMaxPooling1D()(_x)
        convs.append(_x)
    x = Concatenate()(convs)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(
        output_size, activation="relu")(x)
    return x


def image_xception(inputs, output_size):
    cnn = keras.applications.Xception(
        input_tensor=inputs, include_top=False, weights='imagenet')
    x = cnn.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(output_size, activation='relu')(x)
    return x


def image_inception_v3(inputs, output_size):
    cnn = keras.applications.InceptionV3(
        input_tensor=inputs, include_top=False, weights='imagenet')
    x = cnn.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(output_size, activation='relu')(x)
    return x


def image_inception_resnet_v2(inputs, output_size):
    cnn = keras.applications.InceptionResNetV2(
        input_tensor=inputs, include_top=False, weights='imagenet')
    x = cnn.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(output_size, activation='relu')(x)
    return x


def sound_cnn(inputs, output_size):
    def cba(inputs, filters, kernel_size, strides):
        _x = Conv2D(
            filters, kernel_size=kernel_size, strides=strides,
            padding='same')(inputs)
        _x = BatchNormalization()(_x)
        _x = Activation("relu")(_x)
        return _x

    def cba_cluster(x, init_filter=32, init_kernel=(1, 8), init_stride=(1, 2)):
        _x = cba(
            x,
            filters=init_filter,
            kernel_size=init_kernel,
            strides=init_stride)
        _x = cba(
            _x,
            filters=init_filter,
            kernel_size=tuple(reversed(init_kernel)),
            strides=tuple(reversed(init_stride)))
        _x = cba(
            _x,
            filters=init_filter * 2,
            kernel_size=tuple(reversed(init_kernel)),
            strides=tuple(reversed(init_stride)))
        _x = cba(
            _x,
            filters=init_filter * 2,
            kernel_size=tuple(reversed(init_kernel)),
            strides=tuple(reversed(init_stride)))
        return _x

    x_1 = cba_cluster(
        inputs, init_filter=32, init_kernel=(1, 8), init_stride=(1, 2))
    x_2 = cba_cluster(
        inputs, init_filter=32, init_kernel=(1, 16), init_stride=(1, 2))
    x_3 = cba_cluster(
        inputs, init_filter=32, init_kernel=(1, 32), init_stride=(1, 2))
    x_4 = cba_cluster(
        inputs, init_filter=32, init_kernel=(1, 64), init_stride=(1, 2))

    x = Add()([x_1, x_2, x_3, x_4])

    x = cba(x, filters=128, kernel_size=(1, 16), strides=(1, 2))
    x = cba(x, filters=128, kernel_size=(16, 1), strides=(2, 1))

    x = GlobalAveragePooling2D()(x)
    x = Dense(output_size, activation='relu')(x)
    return x
