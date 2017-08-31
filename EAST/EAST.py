#!/bin/env python
"""
EAST.py
Created on 2017/7/14 15:19'
Describe  : 
Athor      : Jin Hou
Email      : 44show@gmail.com
Company : Beijing Linkface co. lot. 
"""
import sys

import tensorflow as tf
import numpy as np

import loss.EAST_loss as east_loss

from netlayers.layer import Layer
import nets.vgg16 as vgg16
import nets.fcn as fcn


class EAST(object):
    def __init__(self, input_layer, task='inference', labels=None, wei_loss=1.0):
        pass
        self.input_layer = input_layer
        self.labels = labels
        self.task = task
        self.wei_loss = wei_loss
        self.layer = Layer()






    def loss(self, canny):
        pass

        with tf.name_scope('loss'):

            score_map_pred = self.inference_map_score()
            if canny:
                score_map_label, rbox_label, quad_label, canny_weight = tf.split(self.labels, [1, 5, 9, 1], 3)
                loss_score_map = east_loss.loss_score_map_with_canny(score_map_pred, score_map_label, canny_weight)
                tf.summary.scalar('loss_score_map_canny', loss_score_map)
            else:
                score_map_label, rbox_label, quad_label = tf.split(self.labels, [1, 5, 9], 3)
                loss_score_map = east_loss.loss_score_map(score_map_pred, score_map_label)
                tf.summary.scalar('loss_score_map', loss_score_map)

            # tf.summary.scalar('loss', score_map_pred)

            rbox_pred = self.inference_rbox()
            # tf.add_to_collection('loss', rbox_pred)

            loss_rbox = east_loss.rbox_aabb_loss(rbox_pred, rbox_label, score_map_label, 10)
            tf.summary.scalar('loss_rbox', loss_rbox)

            # quad_box_pred = self.inference_quad_box()
            # #tf.add_to_collection('loss', quad_box_pred)
            #
            # loss_quad_box = east_loss.quad_loss(quad_box_pred, quad_label, score_map_label)
            # tf.summary.scalar('loss_quad_box', loss_quad_box)

            # loss = loss_score_map + self.wei_loss * (loss_rbox + loss_quad_box)
            loss = loss_score_map + self.wei_loss * (loss_rbox)
            # print(loss_rbox)
            # print(loss_quad_box)
            self.loss = loss
            tf.summary.scalar('final_loss', loss)

        return loss



    def _inference(self, task_name, channel, relu=False ):
        pass

        name = task_name
        bottom_feat_num = self.input_layer.shape[-1].value
        ksize = (1, 1, bottom_feat_num, channel)

        rbox = self.layer._conv_layer(self.input_layer, name=name, ksize=ksize,
                                      top_feat_num=channel, stddev=1.0, relu=relu)

        return rbox

    def inference_rbox(self):
        return self._inference('inference_rbox', 5, relu=True)

    def inference_map_score(self):
        return self._inference('inference_map_score', 1)

    def inference_quad_box(self):
        return self._inference('inference_quad_box', 8)

    # def predict(self):
    #     score_map_pred = self.inference_map_score()
    #     rbox_pred = self.inference_rbox()
    #     quad_box_pred = self.inference_quad_box()
    #
    #     pred = tf.group(score_map_pred, rbox_pred, quad_box_pred)
    #     return pred

    def predict(self):
        score_map_pred = tf.sigmoid(self.inference_map_score())
        rbox_pred = self.inference_rbox()
        # quad_box_pred = self.inference_quad_box()

        #pred = tf.group(score_map_pred, rbox_pred, quad_box_pred)
        return score_map_pred,rbox_pred

