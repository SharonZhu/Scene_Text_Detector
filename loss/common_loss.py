# -*- coding: utf-8 -*-
# @Time     : 2017/8/31  上午11:10
# @Author   : Zhuxinyue_Sharon
# @Email    : zxybuptsee@163.com
# @File     : test.py
# @Software : PyCharm



import tensorflow as tf

def _smooth_l1_loss(box_pred, box_label, box_inside_weights = 1.0, box_outside_weights = 1.0, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2
    print(box_pred.shape, box_label.shape)

    box_diff = box_pred - box_label
    in_box_diff = box_inside_weights * box_diff
    abs_in_box_diff = tf.abs(in_box_diff)
    smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
    in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = box_outside_weights * in_loss_box

    loss_box = tf.reduce_sum(
      out_loss_box,
      axis=dim)
    #print 'xx',out_loss_box.shape, loss_box.shape

    return loss_box


