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
from data.tf_util import write_img_ann_pairs_to_tfrecord

root_path = '../data/'

data_dir = root_path + 'text_data/'
ann_dir = root_path + 'annotation_json/'
filename_pairs = [root_path + 'pair_txt/' + 'img_ann_pair_train.txt',
                  root_path + 'pair_txt/' + 'img_ann_pair_valid.txt',
                  root_path + 'pair_txt/' + 'img_ann_pair_test.txt']

# filename_pairs_test = ['../pair_txt/img_ann_pair_train500.txt']

CANNY = sys.argv[1]  # 'canny or nocanny'
CANNY_BIAS = 0.02

tfrecords_filename = [root_path + 'tfrecords/' + CANNY + '/' + 'train_train.tfrecords',
                      root_path + 'tfrecords/' + CANNY + '/' + 'train_valid.tfrecords',
                      root_path + 'tfrecords/' + CANNY + '/' + 'train_test.tfrecords']

#standard image height and width
stand_img_h = 512
stand_img_w = 512

MODE = 'TRAIN'

if __name__ == '__main__':
    if MODE == 'TRAIN':
        for i in range(3):
            write_img_ann_pairs_to_tfrecord(filename_pairs[i], data_dir, ann_dir, tfrecords_filename[i])
    if MODE == 'OTHER':
        write_img_ann_pairs_to_tfrecord(filename_pairs[2], data_dir, ann_dir, tfrecords_filename[0])