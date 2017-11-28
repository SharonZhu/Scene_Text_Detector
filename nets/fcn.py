#!/bin/env python
"""
fcn.py
Created on 2017/7/12 15:14'
Describe  : use basenet to build FCN layers
Athor      : Jin Hou
Email      : 44show@gmail.com
Company : Beijing Linkface co. lot. 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import math


import numpy as np
import tensorflow as tf

from netlayers.layer import Layer
from nets.vgg16 import VGG16



class FCN(object):
    def __init__(self, base_net):
        self.base_layer_dict = base_net.layer_dict

        self.layer = Layer()

        self.layer_dict = {}
        self.out_layer = None

    """
    def _up_layer(self, bottom, shape,
                       num_classes, name, debug,
                       ksize=4, stride=2):
    """

    def build(self, need_layers, fuse_type='add', up_type='', is_training=True, debug=True):
        pass
        # 'need_layers = [('pool5', 2, out_fea_num, conv1_ksize, conv2_ksize, conv1_out_feat_num, conv2_out_feat_num), ('pool4', 16, out_fea_num)]'

        cur_bottom_layer = None

        for i in range(len(need_layers) - 1):
            base_bottom_layer_name, up_stride_size, up_fea_num, conv1_layer_ksize, conv2_layer_ksize, conv1_out_feat_num, conv2_out_feat_num  = need_layers[i]
            base_top_layer_name = need_layers[i + 1][0]



            if base_bottom_layer_name not in self.base_layer_dict or base_top_layer_name not in self.base_layer_dict:
                print('wrong need_layer: %s, %s'%(base_bottom_layer_name, base_top_layer_name))
                return -1

            # bottom = self.base_layer_dict[base_bottom_layer_name]
            if cur_bottom_layer == None:
                cur_bottom_layer = self.base_layer_dict[base_bottom_layer_name]


            tf.add_to_collection('loss1', cur_bottom_layer)

            base_top_layer = self.base_layer_dict[base_top_layer_name]

            top_shape = tf.shape(base_top_layer)

            up_name = base_bottom_layer_name + '_up'

            ksize = up_stride_size * 2
            print('cur_bottom_layer', cur_bottom_layer)
            print('top_shape',top_shape)


            # cur_up_layer = self.layer._up_layer(cur_bottom_layer, top_shape, up_fea_num, up_name, debug, ksize=ksize, stride=up_stride_size)
            cur_up_layer = self.layer._up_layer_by_resize(cur_bottom_layer, top_shape)
            print('base_top_layer',base_top_layer.shape)
            tf.add_to_collection('loss1', cur_up_layer)



            self.layer_dict[up_name] = cur_up_layer

            concat_name = base_bottom_layer_name + '_concat'
            if fuse_type == 'concat':
                concat_layer = tf.concat([cur_up_layer, base_top_layer],axis=3)
                print('shapes:', cur_up_layer.shape, base_top_layer.shape, concat_layer.shape)
            else:
                pass


            conv_layer_1_name = base_bottom_layer_name + '_conv_1'

            # filter_shape = (conv1_layer_ksize, conv1_layer_ksize,  up_fea_num + base_top_layer.shape[-1].value, conv1_out_feat_num)
            filter_shape = (conv1_layer_ksize, conv1_layer_ksize, concat_layer.shape[-1].value, conv1_out_feat_num)
            print ('filter', filter_shape)

            conv_layer_1 = self.layer._conv_layer(concat_layer, conv_layer_1_name,
                                                  ksize=filter_shape, top_feat_num=conv1_out_feat_num,
                                                  is_training=is_training)
            tf.add_to_collection('loss1', conv_layer_1)

            conv_layer_2_name = base_bottom_layer_name + '_conv_2'

            filter_shape = (conv2_layer_ksize, conv2_layer_ksize, conv1_out_feat_num, conv2_out_feat_num)

            print('filter', filter_shape)

            conv_layer_2 = self.layer._conv_layer(conv_layer_1, conv_layer_2_name,
                                                  ksize=filter_shape, top_feat_num=conv2_out_feat_num,
                                                  is_training=is_training)
            tf.add_to_collection('loss1', conv_layer_2)


            self.layer_dict[conv_layer_2_name] = conv_layer_2

            cur_bottom_layer = conv_layer_2

            print('fcn_layer shape %s:'%conv_layer_2_name, cur_bottom_layer)

        final_in_feat_num = need_layers[-2][-1]
        base_top_layer_name, kise, final_out_feat_num = need_layers[-1]
        final_feat_name = base_top_layer_name + '_final_fcn'
        filter_shape = (ksize, ksize, final_in_feat_num, final_out_feat_num)
        final_layer = self.layer._conv_layer(cur_bottom_layer, final_feat_name,
                                             ksize=filter_shape, top_feat_num=final_out_feat_num,
                                             is_training=is_training)
        self.layer_dict[final_feat_name] = final_layer
        self.out_layer = final_layer


