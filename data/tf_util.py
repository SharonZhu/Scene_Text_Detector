# -*- coding: utf-8 -*-
# @Time     : 2017/8/31  上午11:10
# @Author   : Zhuxinyue_Sharon
# @Email    : zxybuptsee@163.com
# @File     : test.py
# @Software : PyCharm

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
import numpy as np
from scipy import misc
import json
from collections import OrderedDict
import cv2

import data.label_generate as label_generate
import data.image_util as img

#standard image height and width
stand_img_h = 512
stand_img_w = 512

CANNY = False
CANNY_BIAS = 0.01


def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))

def write_img_ann_pairs_to_tfrecord(filename_pairs, data_dir, ann_dir, tfrecords_filename, canny=CANNY):
    '''
    generate tfrecord file
    :param filename_pairs: [txt] img and ann pair path
    :param data_dir: [jpgs] image path
    :param ann_dir: [jsons] ann path
    :param tfrecords_filename: [tfrecords] train, valid, test
    :param random_crop:
    :param crop_num:
    :param canny: if doing canny
    :return:
    '''

    p = 0
    print(filename_pairs)
    fr = open(filename_pairs)
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    for i in fr.readlines():
        print('tfrecording...',p)
        #extract pairs of [img_dir,ann_dir]
        item = i.split(',')

        #read and process image
        image = misc.imread(data_dir + item[0])  # uint8
        image_gray = cv2.imread(data_dir + item[0],0)
        print('original image:',image.shape)

        img_h = image.shape[0] # int
        img_w = image.shape[1]

        if len(image.shape) == 2:
            continue

        fa = open((ann_dir + item[1]).strip('\n'), 'r')
        ann_list = json.load(fa, object_pairs_hook=OrderedDict)
        # print(ann_list)

        if ann_list == []:
            resized_image = np.array(misc.imresize(image, [stand_img_h, stand_img_w]), dtype=np.float32)

            if canny:
                label = np.zeros(shape=[stand_img_h, stand_img_w, 16], dtype=np.float32)
                label[:, :, 15] += CANNY_BIAS
            else:
                label = np.zeros(shape=[stand_img_h, stand_img_w, 15], dtype=np.float32)

            img_raw = resized_image.tostring()
            ann_raw = label.tostring()

            # write to tfRecord
            example = tf.train.Example(features=tf.train.Features(feature={
                'img_raw': _bytes_feature(img_raw),
                'ann_raw': _bytes_feature(ann_raw),
                'img_h': _int64_feature(img_h),
                'img_w': _int64_feature(img_w)
            }))

            writer.write(example.SerializeToString())

        else:
            quad = []
            #ann_quad = np.zeros(shape=[len(ann_list), 8], dtype=np.float32)
            for k in range(len(ann_list)):
                quad.append(ann_list[k]['polygon'])
            quad = np.array(quad,dtype=np.float32)  # [N,8]

            #resize the according quadbox
            print('quad',quad.shape)
            quad = np.reshape(quad,[-1,4,2])
            print('quadxy',quad.shape)

            quad[:,:,0] = (quad[:,:,0] / img_w) * stand_img_w
            quad[:,:,1] = (quad[:,:,1] / img_h) * stand_img_h

            print(quad.shape)
            ann_quad = np.reshape(quad,[-1,8]) #[N,8]

            #resize image to standard height and width
            resized_image = np.array(misc.imresize(image,[stand_img_h,stand_img_w]) , dtype=np.float32)
            print('resized img:', resized_image.shape)
            img_raw = resized_image.tostring()

            if canny:
                resized_image_gray = misc.imresize(image_gray,[stand_img_h,stand_img_w])
                canny_weight = img.canny_generation(resized_image_gray, 21)
                label = label_generate.label_generation_with_canny(stand_img_h, stand_img_w, ann_quad, canny_weight,
                                                               CANNY_BIAS)
                print('label max', label[:, :, 15].max())
            else:
                label = label_generate.label_generation(stand_img_h,stand_img_w,ann_quad,ratio=0.2)

            # display image and polygons
            # from data.image_util import display_image
            # from data.image_util import display_polygons
            # display_image(resized_image, gray=False)
            # display_polygons(quad,'red')

            # label = label_generate.label_generation_with_canny(stand_img_h, stand_img_w, ann_quad, canny,
            #                                                  CANNY_BIAS)

            print('label', label.shape)
            ann_raw = label.tostring()

            #write to tfRecord
            example = tf.train.Example(features = tf.train.Features(feature = {
                'img_raw':_bytes_feature(img_raw),
                'ann_raw':_bytes_feature(ann_raw),
                'img_h':_int64_feature(img_h),
                'img_w':_int64_feature(img_w)
            }))

            writer.write(example.SerializeToString())

        print('label', label.shape)
        p += 1

    writer.close()
    fr.close()