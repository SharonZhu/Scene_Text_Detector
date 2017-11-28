# -*- coding: utf-8 -*-
# @Time     : 2017/8/31  上午11:10
# @Author   : Zhuxinyue_Sharon
# @Email    : zxybuptsee@163.com
# @File     : test.py
# @Software : PyCharm

import sys

import tensorflow as tf
import numpy as nper

import loss.common_loss as com_loss

def rbox_aabb_loss(bottom_pred, labels, select, angle_w = 1.0):
    with tf.name_scope('rbox_aabb_loss') as scope:
        pass
        # box_label, angle_label = tf.split(labels,[4, 1], axis=-1)
        # box_pred, angle_pred = tf.split(bottom_pred,[4, 1], axis=-1)

        def select_func(in_data, select):
            return tf.gather_nd(in_data, select)

        select = tf.stop_gradient(tf.split(tf.where(tf.greater(select, 0.5)), [3, 1], 1)[0])
        print(select.shape)


        select_labels = select_func(labels, select)
        select_pred = select_func(bottom_pred, select)



        box_label, angle_label = tf.split(tf.abs(select_labels), [4, 1], axis=1)
        box_pred, angle_pred = tf.split(select_pred, [4, 1], axis=1)

        tf.add_to_collection('loss', select)
        tf.add_to_collection('loss', select_pred)


        tf.add_to_collection('loss', angle_label)
        tf.add_to_collection('loss', angle_pred)


        delta_angle = angle_label - angle_pred
        tf.add_to_collection('loss', delta_angle)

        #e_delta_angle = tf.reshape(delta_angle, [-1])
        l_angle = tf.reduce_mean(1 - tf.cos(delta_angle))

        tf.add_to_collection('loss',  tf.cos(delta_angle, 'cosres'))

        tf.add_to_collection('loss', l_angle)

        min_dis = tf.minimum(box_label, box_pred)

        def box_area(in_data):
            a, b, c, d = tf.split(in_data, 4, 1)
            return (a + c ) * (b + d)

        with tf.name_scope('iou'):
            area_inters = box_area(min_dis)
            area_pred = box_area(box_pred)
            area_label = box_area(box_label)

            tf.add_to_collection('loss', area_pred)
            tf.add_to_collection('loss', area_label)
            tf.add_to_collection('loss', area_inters)

            iou = area_inters/(area_pred + area_label - area_inters)
            iou = tf.clip_by_value(iou, 1e-10, 512*512)
            tf.add_to_collection('loss', iou)

        l_box = tf.reduce_mean(tf.reshape(-tf.log(iou), [-1]))

        print('iou', iou)
        tf.add_to_collection('loss', l_box)


        return l_box + angle_w * l_angle



def quad_loss(bottom_pred, labels, select):
    pass
    with tf.name_scope('quad_loss') as scope:
        def select_func(in_data, select):
            return tf.gather_nd(in_data, select)

        select = tf.stop_gradient(tf.split(tf.where(tf.greater(select, 0.5)), [3, 1], 1)[0],name='select')
        print(select,'xx',bottom_pred)

        select_pred = select_func(bottom_pred, select)

        #labels = tf.stop_gradient(labels)
        select_labels = select_func(labels, select)

        select_labels = tf.split(select_labels, [2,2,2,2,1], 1)

        min_loss_list = []
        for i in range(4):
            order_vertices_label = tf.concat([select_labels[i % 4], select_labels[(i+1) % 4], select_labels[(i+2) % 4], select_labels[(i+3) % 4]], 1)
            print('xxx', order_vertices_label)
            normalize = select_labels[4] * 8
            print(select_pred.shape, order_vertices_label.shape)

            cur_order_loss = com_loss._smooth_l1_loss(select_pred, order_vertices_label, box_outside_weights=normalize)
            cur_order_loss = tf.reshape(cur_order_loss, [-1,1])
            min_loss_list.append(cur_order_loss)

        min_loss_list = tf.concat(min_loss_list, 1, name='min_loss_list')
        min_loss = tf.reduce_min(min_loss_list, axis=1, name='min_loss')
        min_loss = tf.reduce_mean(min_loss)

        #tf.add_to_collection('loss', min_loss)

        return min_loss







def loss_score_map(pixel_pred,pixel_targets):
    '''
    calculate loss for score map
    :param pixel_pred: predicted score map tensor: [batch_size,img_h,img_w,channel]
    :param pixel_true: true ground truth map tensor: [batch_size,img_h,img_w,channel]
    :return: tensor loss  tf.float
    '''
    with tf.variable_scope('loss') as scope:
        shape_list = pixel_pred.get_shape().as_list()
        print('xxxxx', shape_list)

        shape_num = shape_list[1] * shape_list[2] * shape_list[3]

        pixel_pred= tf.reshape(pixel_pred, [-1, shape_num])   #2D tensor [batch_size,X]
        pixel_targets = tf.reshape(pixel_targets, [-1, shape_num])


        beta = tf.reshape(1 - (tf.reduce_sum(pixel_targets, 1)) / (shape_num), [-1,1]) * 0.98
        tf.add_to_collection('loss', pixel_pred)
        tf.add_to_collection('loss', pixel_targets)
        tf.add_to_collection('loss', beta)
        sig_pred = tf.clip_by_value(tf.sigmoid(pixel_pred), 1.0e-6, 0.99999)

        ls = -beta * (pixel_targets * tf.log(sig_pred)) - (1 - beta) * (1 - pixel_targets) * tf.log(1 - sig_pred)

        # loss_score = tf.reduce_mean(tf.reduce_sum(ls, 1))
        loss_score = tf.reduce_sum(tf.reduce_mean(ls))

        # pos_p = tf.where(tf.greater(pixel_targets, 0.0))
        # # pos_ = tf.gather_nd(ls, pos_select)
        # neg_p = tf.where(tf.less_equal(pixel_targets, 0.0))
        # # neg_ = tf.gather_nd(ls, neg_select)
        # def tmf(x, sel):
        #     return tf.gather_nd(x, sel)
        '''
        tf.add_to_collection('loss', tmf(pixel_pred, pos_p))
        tf.add_to_collection('loss', tmf(sig_pred, pos_p))
        tf.add_to_collection('loss', tmf(tf.log(sig_pred), pos_p))
        tf.add_to_collection('loss', tmf(-beta * (pixel_targets * tf.log(sig_pred)), pos_p))
        tf.add_to_collection('loss', tmf(-(1 - beta) * (1 - pixel_targets) * tf.log(1 - sig_pred), pos_p))
        tf.add_to_collection('loss', tmf(pixel_pred, pos_p))
        tf.add_to_collection('loss', tmf(pixel_pred, neg_p))
        tf.add_to_collection('loss', tmf(sig_pred, neg_p))
        tf.add_to_collection('loss', tmf(tf.log(sig_pred), neg_p))
        tf.add_to_collection('loss', tmf(-beta * (pixel_targets * tf.log(sig_pred)), neg_p))
        tf.add_to_collection('loss', tmf(-(1 - beta) * (1 - pixel_targets) * tf.log(1 - sig_pred), neg_p))
        # tf.add_to_collection('loss', ls)
        # loss_score = tf.reduce_sum(tf.reduce_mean(ls))
        tf.add_to_collection('loss', tf.reduce_sum(tf.reduce_mean(tmf(ls, pos_p))))
        tf.add_to_collection('loss', tf.reduce_sum(tf.reduce_mean(tmf(ls, neg_p))))
        tf.add_to_collection('loss', loss_score)
        '''
    return loss_score


def loss_score_map_with_canny(pixel_pred, pixel_targets, canny_weight):
    with tf.variable_scope('loss') as scope:
        shape_list = pixel_pred.get_shape().as_list()
        print('xxxxx', shape_list)

        shape_num = shape_list[1] * shape_list[2] * shape_list[3]

        pixel_pred= tf.reshape(pixel_pred, [-1, shape_num])   #2D tensor [batch_size,X]
        pixel_targets = tf.reshape(pixel_targets, [-1, shape_num])
        canny_weight = tf.reshape(canny_weight, [-1, shape_num])

        beta = tf.reshape(1 - (tf.reduce_sum(pixel_targets, 1)) / (shape_num), [-1,1])

        tf.add_to_collection('loss', pixel_pred)
        tf.add_to_collection('loss', pixel_targets)
        tf.add_to_collection('loss', beta)

        sig_pred = tf.clip_by_value(tf.sigmoid(pixel_pred), 1.0e-6, 0.99999)

        ls = -beta * (pixel_targets * tf.log(sig_pred)) - (1 - beta) * (1 - pixel_targets) * tf.log(1 - sig_pred)
        ls = ls * canny_weight
        # loss_score = tf.reduce_mean(tf.reduce_sum(ls, 1))
        loss_score = tf.reduce_sum(tf.reduce_mean(ls))

        # pos_p = tf.where(tf.greater(pixel_targets, 0.0))
        # pos_ = tf.gather_nd(ls, pos_select)
        # neg_p = tf.where(tf.less_equal(pixel_targets, 0.0))
        # neg_ = tf.gather_nd(ls, neg_select)
        # def tmf(x, sel):
        #     return tf.gather_nd(x, sel)
        '''
        tf.add_to_collection('loss', tmf(pixel_pred, pos_p))
        tf.add_to_collection('loss', tmf(sig_pred, pos_p))
        tf.add_to_collection('loss', tmf(tf.log(sig_pred), pos_p))
        tf.add_to_collection('loss', tmf(-beta * (pixel_targets * tf.log(sig_pred)), pos_p))
        tf.add_to_collection('loss', tmf(-(1 - beta) * (1 - pixel_targets) * tf.log(1 - sig_pred), pos_p))
        tf.add_to_collection('loss', tmf(pixel_pred, pos_p))
        tf.add_to_collection('loss', tmf(pixel_pred, neg_p))
        tf.add_to_collection('loss', tmf(sig_pred, neg_p))
        tf.add_to_collection('loss', tmf(tf.log(sig_pred), neg_p))
        tf.add_to_collection('loss', tmf(-beta * (pixel_targets * tf.log(sig_pred)), neg_p))
        tf.add_to_collection('loss', tmf(-(1 - beta) * (1 - pixel_targets) * tf.log(1 - sig_pred), neg_p))
        # tf.add_to_collection('loss', ls)
        # loss_score = tf.reduce_sum(tf.reduce_mean(ls))
        tf.add_to_collection('loss', tf.reduce_sum(tf.reduce_mean(tmf(ls, pos_p))))
        tf.add_to_collection('loss', tf.reduce_sum(tf.reduce_mean(tmf(ls, neg_p))))
        tf.add_to_collection('loss', loss_score)
        '''
    return loss_score


























