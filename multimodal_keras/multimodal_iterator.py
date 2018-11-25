import os
import random
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
import cv2
import re
import unicodedata
from collections import OrderedDict

import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

from multimodal_keras import audio_generator


# random erasing for image
def get_random_eraser(p=0.5,
                      s_l=0.02,
                      s_h=0.4,
                      r_1=0.3,
                      r_2=1 / 0.3,
                      v_l=0,
                      v_h=255):
    def eraser(input_img):
        img_h, img_w, _ = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        c = np.random.uniform(v_l, v_h)
        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser


def image_gen(rotation_range=180,
              width_shift_range=0.2,
              height_shift_range=0.2,
              shear_range=10,
              zoom_range=0.3,
              horizontal_flip=True,
              vertical_flip=True,
              channel_shift_range=5.,
              brightness_range=[0.3, 1.0],
              preprocessing_function=get_random_eraser(v_l=0, v_h=255)):
    return ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        channel_shift_range=channel_shift_range,
        brightness_range=brightness_range,
        preprocessing_function=preprocessing_function)


# convert Japanese characters to unicode
# delete a character randomly with del_rate
def convert_text_to_unicode(x, del_rate=0.001):
    if x == 0:
        return [0]
    else:
        return [ord(_x) for _x in str(x).strip() if random.random() > del_rate]


# reshape
def reshape_text(x, max_length=200, del_rate=0.001):
    _x = convert_text_to_unicode(x, del_rate=del_rate)
    _x = _x[:max_length]
    if len(_x) < max_length:
        _x += ([0] * (max_length - len(_x)))
    return _x


# input data generator
class MultiModalIterator():
    def __init__(self,
                 data_df,
                 target_column,
                 train=True,
                 model_type="multiclassifier",
                 batch_size=8,
                 shuffle=True,
                 imagegen=image_gen(),
                 audiogen=audio_generator.AudioGenerator(
                     melsp=True, augment=True),
                 **params):

        self.data_df = data_df
        self.train = train
        self.target_column = self._validate_target_column(target_column)
        self.model_type = model_type
        self.targets = self._get_target()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_num = len(self.data_df)

        self.imagegen = imagegen
        self.audiogen = audiogen

        self.txt_length = params[
            "txt_length"] if "txt_length" in params else 200
        self.img_shape = params["img_shape"] if "img_shape" in params else (
            299, 299, 3)
        self.snd_freq = params["snd_freq"] if "snd_freq" in params else 128
        self.snd_time = params["snd_time"] if "snd_time" in params else 1723

    def __call__(self):
        while True:
            indexes = self._get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size:(i + 1) *
                                    self.batch_size]
                inputs, targets = self._data_generation(batch_ids)

                yield inputs, targets

    def _validate_target_column(self, target_column):
        if target_column not in self.data_df.columns:
            raise KeyError("{0} not in data_df".format(target_column))
        else:
            return target_column

    def _get_target(self):
        if self.model_type == "multiclassifier":
            return self._to_categorical()
        elif self.model_type == "regressor":
            return self._to_regression()
        elif self.model_type == "binaryclassifier":
            return self._to_binary()

    def _to_categorical(self):
        # to categorical
        return keras.utils.to_categorical(
            self.data_df[self.target_column].values)

    def _to_regression(self):
        return np.array([[x] for x in self.data_df[self.target_column].values])

    def _to_binary(self):
        return np.array([[x] for x in self.data_df[self.target_column].values])

    def _get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            # get shuffled indexes
            np.random.shuffle(indexes)

        return indexes

    def _load_txt(self, x):
        # convert Japanese characters to unicode, with random deletion of del_rate
        if self.train:
            _x = np.array([
                reshape_text(t, max_length=self.txt_length, del_rate=0.001)
                for t in x
            ])
        else:
            _x = np.array([
                reshape_text(t, max_length=self.txt_length, del_rate=0)
                for t in x
            ])
        return _x

    def _load_snd(self, x):
        # load sound data
        _x = np.zeros((self.batch_size, self.snd_freq, self.snd_time))
        for i, p in enumerate(x):
            if p is not None:
                # load from .wav format
                wx, _ = audio_generator.load_wave_data(p)
                _x[i] = self.audiogen.augment_sound(
                    wx) if self.train else self.audiogen.calculate_melsp(wx)
        _x = _x.reshape(self.batch_size, self.snd_freq, self.snd_time, 1)
        return _x

    def _load_img(self, x):
        # load image
        _x = np.zeros((len(x), *self.img_shape))
        for i, p in enumerate(x):
            if p is not None:
                # load only if exists
                _x[i] = np.load(p)["img"]
        _x = _x.astype('float32')
        if self.train:
            # data augmentations
            for i in range(self.batch_size):
                _x[i] = self.imagegen.random_transform(_x[i])
                _x[i] = self.imagegen.standardize(_x[i])
        _x /= 255
        return _x

    def _data_generation(self, batch_ids):
        inputs = []
        num_inputs = []
        for i, c in enumerate(self.data_df.columns):
            if c.endswith("num"):
                num_inputs.append(self.data_df[c].values[batch_ids])
            elif c.endswith("txt") or c.endswith("txt_path"):
                x = self._load_txt(self.data_df[c].values[batch_ids])
                inputs.append(x)
            elif c.endswith("img") or c.endswith("img_path"):
                x = self._load_img(self.data_df[c].values[batch_ids])
                inputs.append(x)
            elif c.endswith("snd") or c.endswith("snd_path"):
                x = self._load_snd(self.data_df[c].values[batch_ids])
                inputs.append(x)
        if len(num_inputs) > 0:
            inputs.append(np.array(num_inputs).T)

        targets = self.targets[batch_ids]
        return inputs, targets


class MultiModalModelIterator(MultiModalIterator):
    def __init__(self,
                 multimodalmodel,
                 data_df,
                 target_column,
                 train=True,
                 batch_size=8,
                 shuffle=True,
                 imagegen=image_gen(),
                 audiogen=audio_generator.AudioGenerator(
                     melsp=True, augment=True),
                 **params):
        super().__init__(
            data_df=data_df,
            target_column=target_column,
            train=train,
            batch_size=batch_size,
            shuffle=shuffle,
            imagegen=imagegen,
            audiogen=audiogen,
            **params)
        self.multimodalmodel = multimodalmodel
        self.target_column = self._get_target_column(target_column)
        self.model_type = self.multimodalmodel.model_type
        self.targets = self._get_target()

    def _get_target_column(self, target_column):
        _target_column = self._validate_target_column(target_column)
        if _target_column != self.multimodalmodel.target_column:
            raise ValueError(
                "target_column {0} is different from multimodalmodel target_column {1}".
                format(_target_column, self.multimodalmodel.target_column))
        else:
            return _target_column

    def _data_generation(self, batch_ids):
        inputs = []
        num_inputs = []

        def _format_branch(format, column):
            if format == "num":
                x = np.array(
                    [[x] for x in self.data_df[column].values[batch_ids]])
                inputs.append(x)
            elif format == "txt" or format == "txt_path":
                x = self._load_txt(self.data_df[column].values[batch_ids])
                inputs.append(x)
            elif format == "img" or format == "img_path":
                x = self._load_img(self.data_df[column].values[batch_ids])
                inputs.append(x)
            elif format == "snd" or format == "snd_path":
                x = self._load_snd(self.data_df[column].values[batch_ids])
                inputs.append(x)

        for k, v in self.multimodalmodel.inputs_dict.items():
            if self.multimodalmodel.modal_layer_dict is not None:
                if isinstance(v["column"], str):
                    if v["column"] in self.multimodalmodel.modal_layer_dict.keys(
                    ):
                        _format_branch(v["format"], v["column"])
                        continue
            if isinstance(v["column"], list):
                if v["format"] == "num":
                    for c in v["column"]:
                        num_inputs.append(self.data_df[c].values[batch_ids])
            else:
                _format_branch(v["format"], v["column"])
        if len(num_inputs) > 0:
            inputs.append(np.array(num_inputs).T)

        targets = self.targets[batch_ids]
        return inputs, targets
