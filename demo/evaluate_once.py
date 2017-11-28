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
import copy
import tensorflow as tf
import numpy as np
import linecache
from scipy import misc

import json
from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import sys
sys.path.append('../')
from nets.vgg16  import VGG16
from nets.fcn import FCN
from EAST import EAST
from demo.nms import *
import data.image_util as img
# from demo.evaluation import Evaluate

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

root_path = '../data/data/'
data_dir = root_path + 'text_data_val/'
ann_dir = root_path + 'annotation_json_val/'
test_pair_dir = root_path + 'pair_txt/img_ann_pair_val.txt'
logs_train_dir_canny = root_path + 'out_model/logs/train/canny/'
logs_train_dir_nocanny = root_path + 'out_model/logs/train/nocanny/'

NUM_TEST = 15
BATCH_SIZE = 1
#standard image height and width
stand_img_h = 512
stand_img_w = 512


def create_ramdom_dir(test_pair_dir):
    '''ramdomly get an image from specific path'''
    item = linecache.getline(filename=test_pair_dir, lineno=np.random.randint(0, NUM_TEST) + 1)
    print('total_dir', item)
    item_split = item.split(',')
    img_dir = data_dir + item_split[0]
    print('image_dir', img_dir)
    anno_dir = (ann_dir + item_split[1]).strip('\n')
    print('ann_dir', anno_dir)

    return img_dir, anno_dir

def get_one_image(img_dir, anno_dir):
    quad_origin = None

    # read and process image
    image = misc.imread(img_dir)  # uint8
    img_h = image.shape[0]  # int
    img_w = image.shape[1]
    print('img_h:',img_h)
    print('img_w:',img_w)
    # image_array = np.array(misc.imresize(image,[stand_img_h,stand_img_w]) , dtype=np.float32)
    image_array = np.array(image,dtype=np.float32)
    print(image.shape)

    fa = open(anno_dir, 'r')
    ann_list = json.load(fa, object_pairs_hook=OrderedDict)
    # print(ann_list)
    quad = []
    for k in range(len(ann_list)):
        quad.append(ann_list[k]['polygon'])
    quad = np.array(quad, dtype=np.float32)  # [N,8]
    quad = np.reshape(quad, [-1, 4, 2])


    img.display_image(image,gray=False)
    img.display_polygons(quad,'red')

    img.display_image(image,gray=False)
    img.display_polygons(quad,'red')

    quad_origin = copy.deepcopy(quad)

    image_array = np.array(misc.imresize(image,[stand_img_h,stand_img_w]) , dtype=np.float32)
    image_array = np.array(image_array)
    quad[:, :, 0] = (quad[: ,:, 0] * stand_img_w) / img_w
    quad[:, :, 1] = (quad[:, :, 1] * stand_img_h) / img_h

    image_array = np.reshape(image_array, [1, stand_img_h, stand_img_w, 3])

    return image_array, quad_origin, quad, img_h, img_w

# get_one_image(test_pair_dir)


def restore_geo(mode, score_map_thresh, input_score, input_box, height, width):
    '''

    :param mode: 'rbox' or 'quad'
    :param input_score: the predicted score map of an image [1,X,X,1]
    :param input_box: the predicted rbox [1,X,X,5] or quad [1,X,X,8]
    :param height: the original height of the image
    :param width: the original width of the iamge
    :return:
    '''

    #input_score = np.reshape(input_score,[row_pixel,line_pixel])
    input_score = input_score[0,:,:,0]
    input_box = input_box[0,:,:,:]

    # print('input_score',input_score)
    # print('input_box', input_box)

    # coord_array = np.where(input_score > 0.65)
    # coord_list = []
    #
    # for i in range(len(coord_array[0])):
    #     coord_list.append(np.array([coord_array[0][i], coord_array[1][i]]))

    coord_list = np.argwhere(input_score >= score_map_thresh)
    print (len(coord_list))
    print(coord_list.shape)

    num_bbox = len(coord_list)
    bbox = np.empty(shape=[num_bbox, 8], dtype=np.float32)
    prob = np.empty(shape=[num_bbox, 1], dtype=np.float32)

    if mode == 'rbox':
        d = input_box[:,:,:4]
        angle = input_box[:,:,4]

        for j in range(num_bbox):

            # if j % 5 !=0:
            #     continue


            # the coord of a picked pixel
            index_0 = coord_list[j][0]
            index_1 = coord_list[j][1]

            prob[j,:] = (input_score[index_0, index_1])

            per_d = d[index_0, index_1, :]

            pixelx = index_1
            pixely = index_0

            if angle[index_0, index_1] <= 45:
                per_bbox_x = np.array([[pixelx - per_d[3]], [pixelx + per_d[1]], [pixelx + per_d[1]], [pixelx - per_d[3]]],
                                         dtype=np.float32)
                per_bbox_y = np.array([[pixely - per_d[0]], [pixely - per_d[0]], [pixely + per_d[2]], [pixely + per_d[2]]],
                                         dtype=np.float32)

                # resize the according quadbox
                per_bbox_x = (per_bbox_x / stand_img_w) * width
                per_bbox_y = (per_bbox_y / stand_img_h) * height

                per_bbox = np.concatenate((per_bbox_x,per_bbox_y),axis=1)

                # current_axis = plt.gca()
                # current_axis.add_patch(Polygon(xy=per_bbox, linewidth=0.2, alpha=1, fill=None, edgecolor='yellow'))

                # per_bbox_x_p = (pixelx / stand_img_w) * width
                # per_bbox_y_p = (pixely / stand_img_w) * height

                # current_axis.scatter(per_bbox_x_p, per_bbox_y_p, color='green')

                per_bbox = np.reshape(per_bbox, [1, 8])

                bbox[j, :] = per_bbox

        concat_bbox = np.concatenate((bbox, prob), axis=1)
        print(concat_bbox.shape)

        if concat_bbox.shape[0] == 0:
            nms_bbox = np.array([])
        else:
            nms_bbox = nms_locality(concat_bbox, thres=0.25)
            print(nms_bbox.shape)
            nms_bbox = nms_bbox[:, :8]
            nms_bbox = np.reshape(nms_bbox, [-1, 4, 2])
            print(nms_bbox.shape)

        # display_polygons(nms_bbox, 'yellow')

    if mode == 'quad':
        for j in range(num_bbox):
            # the coord of a picked pixel
            pixelx = coord_list[j][0]
            pixely = coord_list[j][1]

            per_bbox = np.reshape(bbox[j,:],[4,2])
            per_pixel = np.reshape(input_box[pixelx,pixely,:],[4,2])

            per_bbox[:, 0] =  per_pixel[:,0] + pixelx
            per_bbox[:, 1] = per_pixel[:, 1] + pixely

            # resize the according quadbox
            per_bbox[:, 0] = (per_bbox[:, 0] / stand_img_w) * width
            per_bbox[:, 1] = (per_bbox[:, 1] / stand_img_h) * height

            current_axis = plt.gca()
            current_axis.add_patch(Polygon(xy=per_bbox,linewidth=0.02,alpha=1,fill=None,edgecolor='yellow'))

            per_bbox = np.reshape(per_bbox,[1,8])

            bbox[j,:] = per_bbox

    return bbox, nms_bbox

# test restore_geo()
# input_score = np.ones(shape=[1,2,2,1],dtype=np.float32)
# input_box = np.array([[0,0,8,0,9,2,1,2],[0,0,8,0,9,2,1,2],[0,0,8,0,9,2,1,2],[0,0,8,0,9,2,1,2]])
# input_box = np.reshape(input_box,[1,2,2,8])
# print(input_box)
# print(input_box.shape)
# height = 1
# width = 1
# bbox = restore_geo(mode='quad',input_score=input_score,input_box=input_box,height=height,width=width)
# print(bbox)
# print(bbox.shape)

def detect_one_image(img_dir, anno_dir, logs_train_dir):
    image_array, quad_origin, quad, img_h, img_w = get_one_image(img_dir, anno_dir)
    print("h and w", img_h,img_w)
    print('logs_dir', logs_train_dir)

    with tf.Graph().as_default():
        # image = tf.cast(image_array, tf.float32)
        rgb = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE,stand_img_h, stand_img_w, 3])

        my_vgg16 = VGG16(root_path + 'preNETS/vgg16.npy')

        with tf.name_scope('netVgg16') as netVgg16:
            my_vgg16.build(rgb, is_training=False)

        my_fcn = FCN(my_vgg16)
        with tf.name_scope('netFCN') as netFCN:
            '''
            'need_layers = [('pool5', stride, up_fea_num, conv1_ksize, conv2_ksize, conv1_out_feat_num, conv2_out_feat_num), ('pool4', 16, out_fea_num)]'
            '''
            # my_fcn.build(
            #     [('pool5', 2, 64, 1, 3, 64, 64), ('pool4', 2, 64, 1, 3, 64, 64), ('pool3', 2, 32, 1, 3, 32, 32),
            #      ('pool2', 4, 32, 1, 3, 32, 32), ('bgr', 3, 32)], fuse_type='concat', debug=False)
            my_fcn.build(
                [('pool5', 2, 64, 1, 3, 128, 128), ('pool4', 2, 64, 1, 3, 96, 64), ('pool3', 2, 32, 1, 3, 64, 32),
                 ('pool2', 4, 32, 1, 3, 32, 32), ('bgr', 3, 32)], fuse_type='concat', debug=False, is_training=False)

        my_east = EAST.EAST(my_fcn.out_layer, task='train', is_training=False)
        predict_score, predict_rbox = my_east.predict()
        print('predict:', predict_score, predict_rbox)
        # predict_score = my_east.predict()
        # print('predict:',predict_score)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            print(logs_train_dir)

            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print(global_step)
                print(ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("successful loading, global step is %s" %global_step)

            else:
                print("no checkpoint file founded")
                return

            pred_s,pred_r = sess.run([predict_score,predict_rbox], feed_dict={rgb:image_array})
            # print('predict score', pred_s)
            # print(pred_q)
            pred_bbox_rbox, pred_nms_rbox = restore_geo(mode='rbox',score_map_thresh=0.95, input_score=pred_s,input_box=pred_r,
                                    height=img_h,width=img_w)

            # if you want to evaluate one image:
            '''
            precision, recall, fscore = Evaluate(quad_origin, pred_nms_rbox, 0.5)
            print('precision', precision)
            print('recall', recall)
            print('fscore', fscore)
            '''
            # pred_bbox_quad = restore_geo(mode='quad', score_map_thresh=0.65, input_score=pred_s, input_box=pred_q,
            #                         height=img_h, width=img_w)
            print('predict rbox', pred_bbox_rbox)

def main(argv=None):  # pylint: disable=unused-argument
    img_dir, anno_dir = create_ramdom_dir(test_pair_dir)
    detect_one_image(img_dir, anno_dir, logs_train_dir=logs_train_dir_nocanny)
    detect_one_image(img_dir, anno_dir, logs_train_dir=logs_train_dir_canny)
    plt.show()

if __name__ == '__main__':
  tf.app.run()



