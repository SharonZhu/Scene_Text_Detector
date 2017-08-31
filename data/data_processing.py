# -*- coding: utf-8 -*-
# @Time     : 2017/8/31  上午11:10
# @Author   : Zhuxinyue_Sharon
# @Email    : zxybuptsee@163.com
# @File     : test.py
# @Software : PyCharm

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from data.tf_util import write_img_ann_pairs_to_tfrecord

data_root_path = '/Users/zhuxinyue/Documents/EAST/'
tf_root_path = 'tfrecords/'

data_dir = data_root_path + 'text_data/'
ann_dir = data_root_path + 'annotation_json/'

filename_pairs = ['pair_txt/img_ann_pair_train.txt',
                  'pair_txt/img_ann_pair_valid.txt',
                  'pair_txt/img_ann_pair_test.txt']
filename_pairs_test = ['../pair_txt/img_ann_pair_train500.txt']

tfrecords_filename_train = tf_root_path + 'train_train.tfrecords'
tfrecords_filename_valid = tf_root_path + 'train_valid.tfrecords'
tfrecords_filename_test = tf_root_path + 'train_test.tfrecords'
tfrecords_filename = [tfrecords_filename_train,tfrecords_filename_valid,tfrecords_filename_test]

MODE = 'TRAIN'

if __name__ == '__main__':
    if MODE == 'TRAIN':
        for i in range(3):
            write_img_ann_pairs_to_tfrecord(filename_pairs[i], data_dir, ann_dir, tfrecords_filename[i],
                                            random_crop=False, crop_num=None)
    if MODE == 'OTHER':
        write_img_ann_pairs_to_tfrecord(filename_pairs_test, data_dir, ann_dir, tfrecords_filename[0],
                                        random_crop=False, crop_num=None)