# -*- coding: utf-8 -*-
# @Time     : 2017/8/31  上午11:10
# @Author   : Zhuxinyue_Sharon
# @Email    : zxybuptsee@163.com
# @File     : test.py
# @Software : PyCharm


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import math
import sys

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from netlayers.layer import Layer

VGG_MEAN = [103.939, 116.779, 123.68]


class VGG16(object):

    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = sys.modules[self.__class__.__module__].__file__
            # print path
            path = os.path.abspath(os.path.join(path, os.pardir))
            # print path
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            logging.info("Load npy file from '%s'.", vgg16_npy_path)
        if not os.path.isfile(vgg16_npy_path):
            logging.error(("File '%s' not found. Download it from "
                           "ftp://mi.eng.cam.ac.uk/pub/mttt2/"
                           "models/vgg16.npy"), vgg16_npy_path)
            sys.exit(1)

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        self.wd = 5e-4

        print("npy file loaded")

        self.layer = Layer(self.data_dict, self.wd)

        # save all layer for fcn
        self.layer_dict = {}

    def build(self, rgb, is_training=True, need_fc=False, train=False, num_classes=20, random_init_fc8=False,
              debug=False):
        """
        Build the VGG model using loaded weights
        Parameters
        ----------
        rgb: image batch tensor
            Image in rgb shap. Scaled to Intervall [0, 255]
        train: bool
            Whether to build train or inference graph
        num_classes: int
            How many classes should be predicted (by fc8)
        random_init_fc8 : bool
            Whether to initialize fc8 layer randomly.
            Finetuning is required in this case.
        debug: bool
            Whether to print additional Debug Information.
        """
        # Convert RGB to BGR
        print('is_training', is_training)
        with tf.name_scope('Processing'):
            # rgb = tf.image.convert_image_dtype(rgb, tf.float32)
            red, green, blue = tf.split(rgb, 3, 3)
            # assert red.get_shape().as_list()[1:] == [224, 224, 1]
            # assert green.get_shape().as_list()[1:] == [224, 224, 1]
            # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
            bgr = tf.concat([
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2]], axis=3)

            if debug:
                bgr = tf.Print(bgr, [tf.shape(bgr)],
                               message='Shape of input image: ',
                               summarize=4, first_n=1)

        self.conv1_1 = self.layer._conv_layer(bgr, "conv1_1", is_training=is_training)
        self.layer_dict["conv1_1"] = self.conv1_1
        self.conv1_2 = self.layer._conv_layer(self.conv1_1, "conv1_2", is_training=is_training)
        self.layer_dict["conv1_2"] = self.conv1_2
        self.pool1 = self.layer._max_pool(self.conv1_2, 'pool1', debug)
        self.layer_dict["bgr"] = bgr
        self.layer_dict["pool1"] = self.pool1

        self.conv2_1 = self.layer._conv_layer(self.pool1, "conv2_1", is_training=is_training)
        self.conv2_2 = self.layer._conv_layer(self.conv2_1, "conv2_2", is_training=is_training)
        self.pool2 = self.layer._max_pool(self.conv2_2, 'pool2', debug)
        self.layer_dict["pool2"] = self.pool2

        self.conv3_1 = self.layer._conv_layer(self.pool2, "conv3_1", is_training=is_training)
        self.conv3_2 = self.layer._conv_layer(self.conv3_1, "conv3_2", is_training=is_training)
        self.conv3_3 = self.layer._conv_layer(self.conv3_2, "conv3_3", is_training=is_training)
        self.pool3 = self.layer._max_pool(self.conv3_3, 'pool3', debug)
        self.layer_dict["pool3"] = self.pool3

        self.conv4_1 = self.layer._conv_layer(self.pool3, "conv4_1", is_training=is_training)
        self.conv4_2 = self.layer._conv_layer(self.conv4_1, "conv4_2", is_training=is_training)
        self.conv4_3 = self.layer._conv_layer(self.conv4_2, "conv4_3", is_training=is_training)
        self.pool4 = self.layer._max_pool(self.conv4_3, 'pool4', debug)
        self.layer_dict["pool4"] = self.pool4

        self.conv5_1 = self.layer._conv_layer(self.pool4, "conv5_1", is_training=is_training)
        self.conv5_2 = self.layer._conv_layer(self.conv5_1, "conv5_2", is_training=is_training)
        self.conv5_3 = self.layer._conv_layer(self.conv5_2, "conv5_3", is_training=is_training)
        self.pool5 = self.layer._max_pool(self.conv5_3, 'pool5', debug)
        self.layer_dict["pool5"] = self.pool5

        """
        self.fc6 = self.layer._fc_layer(self.pool5, "fc6")
        self.layer_dict["fc6"] = self.fc6

        if train:
            self.fc6 = tf.nn.dropout(self.fc6, 0.5)

        self.fc7 = self.layer._fc_layer(self.fc6, "fc7")
        self.layer_dict["fc7"] = self.fc7

        if train:
            self.fc7 = tf.nn.dropout(self.fc7, 0.5)

        if random_init_fc8:
            self.score_fr = self.layer._score_layer(self.fc7, "score_fr",
                                              num_classes)
        else:
            self.score_fr = self.layer._fc_layer(self.fc7, "score_fr",
                                           num_classes=num_classes,
                                           relu=False)
        self.layer_dict["score_fr"] = self.score_fr
        self.pred = tf.argmax(self.score_fr, dimension=3)

        """






if __name__ == "__main__":
    print('test vgg16')
    my_vgg16 = VGG16('../../data/pre_NETS/vgg16.npy')
    rgb = tf.placeholder(dtype=tf.float32, shape=[12, 512, 512, 3])
    with tf.name_scope('testVgg16') as testVgg16:
        my_vgg16.build(rgb, is_training=False)