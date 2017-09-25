# -*- coding: utf-8 -*-
# @Time     : 2017/8/31  上午11:10
# @Author   : Zhuxinyue_Sharon
# @Email    : zxybuptsee@163.com
# @File     : test.py
# @Software : PyCharm

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import data.geo_util as geo
import cv2
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt

def pixel_geo_generation(pixelx, pixely, min_area_rect, rect_box, quad):
    '''
    provide rbox or quad label
    :param pixelx/y: coord of a pixel to calculate
    :param rbox: [x,y,height,width]
    :param quad: [xi,yi] 4*2
    :return: label_rbox 5 + label_quad 8
    '''
    #rbox generation
    label_rbox = np.empty([1,5],dtype=np.float32)

    for i in range(4):
        # print(rect_box[i],rect_box[(i+1)%4], (pixely, pixelx))
        label_rbox[0,i] = geo.p_to_l_dist(rect_box[i],rect_box[(i+1)%4], (pixely, pixelx))
        #four distances towards bottom, left, top, right
    label_rbox[0,4] = 90 + min_area_rect[2] #angle

    #quad generation
    label_quad = np.zeros([4,2])
    label_quad[:,0] = quad[:,0] - pixelx
    label_quad[:,1] = quad[:,1] - pixely
    label_quad = np.reshape(label_quad,(1,8))

    return label_rbox,label_quad

def label_generation(row_pixel,line_pixel,quad_gt_in,ratio):
    '''
    given the size of an image, generate labels
    :param row_pixel: num of pixels in a row
    :param line_pixel: num of pixels in a line
    :param quad_gt: a list of ground_truth quad (may be more than one)
    :param ratio: ratio of reference
    :return:
    '''
    pixel_set = np.zeros([row_pixel,line_pixel,15],dtype=np.float32)
    num_quad = quad_gt_in.shape[0]  #number of quad in an image

    for k in range(num_quad):
        print('processing...', k)
        quad_gt = np.reshape(quad_gt_in[k, :], [4, 2])

        # Calculate min_area_rect
        min_area_rect = cv2.minAreaRect(quad_gt)
        # ((centerx,centery), (height, width), angle)
        rect_box = cv2.boxPoints(min_area_rect)
        rect_box = np.round(rect_box)  # for points of the rect(right-top,anti-clockwise)
        arg_min_xy = np.argmin(rect_box, axis=0)

        arg_min_y = arg_min_xy[1]
        rect_box = np.array([rect_box[arg_min_y], rect_box[(arg_min_y + 1) % 4], rect_box[(arg_min_y + 2) % 4],
                             rect_box[(arg_min_y + 3) % 4]])
        # print(rect_box)

        quad_gt_sh18, quad_gt_sh42 = geo.shrink_quad(quad_gt, ratio=ratio)

        # test quad:
        # import matplotlib.pyplot as plt
        # from matplotlib.patches import Polygon
        # current_axis = plt.gca()
        # current_axis.add_patch(Polygon(xy=quad_gt_sh42, linewidth=1, alpha=1, fill=None, edgecolor='yellow'))

        for i in range(row_pixel):
            for j in range(line_pixel):
                    if (cv2.pointPolygonTest(contour=quad_gt_sh42, pt=(j, i), measureDist=False)) == 1:

                        pixel_set[i ,j, 0] = 1
                        pixel_set[i, j, 1:6], pixel_set[i, j, 6:14] = pixel_geo_generation(i, j,
                                                                                      min_area_rect=min_area_rect,
                                                                                      rect_box=rect_box,
                                                                                      quad=quad_gt)
                        pixel_set[i,j,14] = geo.min_line_for_quad(quad_gt)

    return pixel_set

def label_generation_with_canny(row_pixel, line_pixel, quad_gt_in, canny_weight, bias):

    pixel_set = np.zeros([row_pixel, line_pixel, 16], dtype=np.float32)
    num_quad = quad_gt_in.shape[0]  # number of quad in an image

    print(quad_gt_in)

    for k in range(num_quad):
        print('processing...', k)
        quad_gt = np.reshape(quad_gt_in[k, :], [4, 2])


        # Calculate min_area_rect
        min_area_rect = cv2.minAreaRect(quad_gt)
        # ((centerx,centery), (height, width), angle)
        rect_box = cv2.boxPoints(min_area_rect)
        rect_box = np.round(rect_box)  # for points of the rect(right-top,anti-clockwise)
        arg_min_xy = np.argmin(rect_box, axis=0)

        arg_min_y = arg_min_xy[1]
        rect_box = np.array([rect_box[arg_min_y], rect_box[(arg_min_y + 1)%4], rect_box[(arg_min_y + 2)%4],
                             rect_box[(arg_min_y + 3)%4]])
        # print(rect_box)

        # test quad:
        # current_axis = plt.gca()
        # current_axis.add_patch(Polygon(xy=quad_gt, linewidth=1, alpha=1, fill=None, edgecolor='yellow'))

        for i in range(row_pixel):
            for j in range(line_pixel):
                if (cv2.pointPolygonTest(contour=quad_gt, pt=(j, i), measureDist=False)) == 1:

                    pixel_set[i, j, 0] = 1
                    pixel_set[i, j, 1:6], pixel_set[i, j, 6:14] = pixel_geo_generation(i, j,
                                                                                       min_area_rect=min_area_rect,
                                                                                       rect_box=rect_box,
                                                                                       quad=quad_gt)
                    pixel_set[i, j, 14] = geo.min_line_for_quad(quad_gt)
                    pixel_set[i, j, 15] = canny_weight[i, j] * 2 + bias * 2
                elif pixel_set[i, j, 0] != 1:
                    pixel_set[i, j, 15] = max(0, canny_weight[i, j] * 3)

    pixel_set[:, :, 15] += bias

    return pixel_set