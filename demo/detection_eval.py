# -*- coding: utf-8 -*-
# @Time     : 2017/10/10  上午10:49
# @Author   : Zhuxinyue_Sharon
# @Email    : zxybuptsee@163.com
# @File     : detection_eval.py
# @Software : PyCharm

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import cv2
from collections import defaultdict, namedtuple, OrderedDict
import tensorflow as tf
from scipy import misc
# import matplotlib.pyplot as plt
# from itertools import count
from bisect import bisect
import json

# sys.path.append('../')
from nets.vgg16  import VGG16
from nets.fcn import FCN
from EAST import EAST
from demo.evaluate_once import restore_geo


root_path = '../data/data/'

data_dir = root_path + 'text_data/'
ann_dir = root_path + 'annotation_json/'
testdir = root_path + 'pair_txt/' + 'img_ann_pair_train15.txt'
logs_train_dir = root_path + 'out_model/logs/canny_20k/'

stand_img_h = 512
stand_img_w = 512

Circle = namedtuple('Circle', 'center, radius')


def find_best_circle(poly):
    """ poly: a list of 4 points
        return a circle in ((x, y), r) , who has a largest IOU with poly
    """
    def get_poly_points(poly):
        PADDING = 10
        FILL = 255
        poly = np.asarray(poly, dtype=np.int32)
        maxx, maxy = max([k[0] for k in poly]), max([k[1] for k in poly])
        minx, miny = min([k[0] for k in poly]), min([k[1] for k in poly])
        img = np.zeros((maxy + PADDING, maxx + PADDING), dtype='uint8')
        cv2.fillConvexPoly(img, poly, FILL)

        points = []
        for x in range(minx, maxx + 1):
            for y in range(miny, maxy + 1):
                if img[y, x] == FILL:
                    points.append((x, y))
        return np.asarray(points)

    def get_com(points):
        return np.round(np.mean(points, axis=0))

    def find_best_radius(point, points):
        def dist(p):
            return (point[0] - p[0]) ** 2 + (point[1] - p[1]) ** 2
        dists = sorted([dist(p) for p in points])
        maxr = int(np.sqrt(dists[-1])) + 1
        best_iou = 0.0
        best_r = 0
        for r in range(maxr):
            idx = bisect(dists, r * r)  # i
            circle_area = np.pi * r * r
            union = len(points) + circle_area - idx
            iou = float(idx) / union
            if iou > best_iou:
                best_iou = iou
                best_r = r
        return best_r, best_iou

    points = get_poly_points(poly)

    best_p, best_r, best_iou = None, None, 0.0
    best_p = get_com(points)
    best_r, best_iou = find_best_radius(best_p, points)
    #for p in points:
        #r, iou = find_best_radius(p, points)
        #if iou > best_iou:
            #best_iou = iou
            #best_p = p
            #best_r = r
    return Circle(best_p, best_r)
#     return best_p, best_r, best_iou

def eval_circles(img_shape, gt, pred):
    gt_img = np.zeros(img_shape, dtype='uint8')
    pred_img = np.zeros(img_shape, dtype='uint8')

    for c in gt:
        cv2.circle(gt_img, tuple(map(int, c.center)), int(c.radius), 255,
                   thickness=-1)
    for c in pred:
        cv2.circle(pred_img, tuple(map(int, c.center)), int(c.radius), 255,
                   thickness=-1)

    gt_cnt = (gt_img > 0).sum()
    pred_cnt = (pred_img > 0).sum()
    corr = ((gt_img > 0) * (pred_img > 0)).sum()

    def score(tot):
        if tot == 0:
            return 1.0
        else:
            return corr / float(tot)

    recall = score(gt_cnt)
    precision = score(pred_cnt)
    return precision, recall

def read_test_set(data_path):
    p = 0
    fr = open(data_path)
    images = []
    label = []
    labels = []
    heights = []
    widths = []

    for i in fr.readlines():
        p += 1
        if p % 10 == 0:
            print('Reading...', p)

        item = i.split(',')
        img_dir = data_dir + item[0]
        anno_dir = ann_dir + item[1]

        image = misc.imread(img_dir)  # uint8

        img_h = image.shape[0]
        img_w = image.shape[1]

        image = misc.imresize(image, [512, 512])
        image = np.reshape(image, [512, 512, -1])
        if image.shape[2] == 1:
            continue
        images.append(image)
        heights.append(img_h)
        widths.append(img_w)

        fa = open((anno_dir).strip('\n'), 'r')
        ann_list = json.load(fa, object_pairs_hook=OrderedDict)

        for k in range(len(ann_list)):
            label.append(ann_list[k]['polygon'])
        label = np.array(label, dtype=np.float32)  # [N,8]

        labels.append(label)
        label = []
    images = np.array(images, dtype=np.float32)
    heights = np.array(heights, dtype=np.int64)
    widths = np.array(widths, dtype=np.int64)
    # labels = np.array(labels, dtype=np.float32)

    return images, labels, heights, widths

# a, b, c, d = read_test_set(root_path + 'pair_txt/' + 'img_ann_pair_train500.txt')
# print(a.shape, type(b), c.shape, d.shape)
# print(b[0].shape, c[0], d[0])

class Stat(object):
    def __init__(self):
        self.precision_sum = 0.0
        self.recall_sum = 0.0
        self.count = 0

    def add(self, precision, recall):
        self.count += 1
        self.precision_sum += precision
        self.recall_sum += recall

    @property
    def precision(self):
        if self.count == 0:
            return 1.0
        return self.precision_sum / self.count

    @property
    def recall(self):
        if self.count == 0:
            return 1.0
        return self.recall_sum / self.count

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r)

def get_pred_polys(test_image, test_height, test_width, sess, rgb, predict_score, predict_rbox):
    test_image = np.reshape(test_image, [1, 512, 512, 3])
    # display_image(origin_image)
    # display_polygons(test_labels[step], 'red')

    pred_s, pred_r = sess.run([predict_score, predict_rbox], feed_dict={rgb: test_image})
    # print('predict score', pred_s)
    # print('label', test_labels[step].shape)
    pred_bbox_rbox, pred_nms_rbox = restore_geo(mode='rbox', score_map_thresh=0.99999,
                                                input_score=pred_s, input_box=pred_r,
                                                height=test_height, width=test_width)

    return pred_nms_rbox

def detections(test_pair_path, logs_train_dir):
    stat = Stat()
    circle_gt_polys = []
    circle_pred_polys = []

    images, labels, heights, widths = read_test_set(test_pair_path)

    with tf.Graph().as_default():
        rgb = tf.placeholder(dtype=tf.float32, shape=[1, stand_img_h, stand_img_w, 3])
        my_vgg16 = VGG16(root_path + 'preNETS/vgg16.npy')

        with tf.name_scope('netVgg16') as netVgg16:
            my_vgg16.build(rgb)

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
                 ('pool2', 4, 32, 1, 3, 32, 32), ('bgr', 3, 32)], fuse_type='concat', debug=False)

        my_east = EAST.EAST(my_fcn.out_layer, task='train')
        predict_score, predict_rbox = my_east.predict()

        # print('predict:', predict_score, predict_rbox)
        # predict_score = my_east.predict()
        # print('predict:',predict_score)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            sess.run(init_op)

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            print(logs_train_dir)

            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print(global_step)
                print(ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("successful loading, global step is %s" % global_step)

            else:
                print("no checkpoint file founded")
                return

            for i in range(images.shape[0]):
                # from demo.evaluate_once import display_image
                # from demo.evaluate_once import display_polygons

                image = images[i]
                height = heights[i]
                width = widths[i]
                shape = [height, width]
                gt_polys = labels[i]

                # display image
                # image_disp = misc.imresize(image, [height, width])
                # display_image(image_disp)
                # display_polygons(gt_polys, 'red')

                # get gt circles
                for poly in gt_polys:
                    poly = np.reshape(poly, [4, 2])
                    circle_gt_polys.append(find_best_circle(poly))
                print('ground_truth', gt_polys)
                print(circle_gt_polys)


                # get predict circles
                pred_polys = get_pred_polys(image, height, width, sess, rgb, predict_score, predict_rbox)
                # display_polygons(pred_polys, 'green')

                for poly in pred_polys:
                    poly = np.reshape(poly, [4, 2])
                    circle_pred_polys.append(find_best_circle(poly))

                print('predict', pred_polys.shape)
                print(circle_pred_polys)

                # Calculate precision and recall for an image
                stat.add(*eval_circles(shape, circle_gt_polys, circle_pred_polys))

                print('[%d] image, p[%f], r[%f], f1[%f]' % (i, stat.precision, stat.recall, stat.f1))

    print(stat.precision, stat.recall, stat.f1)
    sess.close()


def main(argv=None):  # pylint: disable=unused-argument
    detections(testdir, logs_train_dir)

if __name__ == '__main__':
  tf.app.run()



















