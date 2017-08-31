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
import cv2

def distL2(a,b):
    '''
    :param a: ndarray
    :param b: ndarray
    :return: L2 distance between vector a and b
    '''
    return np.sqrt(np.sum((a-b) ** 2))

def shrink_line(cord1, cord2, ratio, r):
    '''
    shrink a line with cord1 and cord2 of ratio r
    :param cord1:
    :param cord2:
    :param ratio:
    :param r: reference
    :return: shrinked cord
    '''
    shrink = np.array((0,0),dtype=np.float32)
    L = distL2(cord1, cord2)
    a = ratio*r
    if cord1[1] > cord2[1] or (cord1[1]==cord2[1] and cord1[0]>cord2[0]):
        a = L-a
    shrink[0] = round(abs((cord2[0] - cord1[0])) * (a / L) + min(cord1[0],cord2[0]))
    shrink[1] = round(abs((cord2[1] - cord1[1])) * (a / L) + min(cord1[1],cord2[1]))
    return shrink

def min_line_for_quad(quad):
    min_line1 = min(distL2(quad[0,:],quad[1,:]),distL2(quad[1,:],quad[2,:]))
    min_line2 = min(distL2(quad[2, :],quad[3, :]),min_line1)
    min_line = min(distL2(quad[3, :],quad[0, :]),min_line2)
    return min_line


def shrink_quad(quad_gt,ratio):
    '''
    shrink ground_truth box
    :param quad_gt: [xi,yi] of gt_box, ndarray:4*2
    :return: shrink:1*8; shrink_short:4*2
    '''
    # quad_gt_shrink = np.empty([4,2],dtype=float)
    # quad_gt_forward = np.roll(quad_gt,6)
    # quad_gt_backward = np.roll(quad_gt,2)

    shrink_long = np.empty([4,2],dtype=np.float32)
    shrink_short = np.empty([4,2],dtype=np.float32)
    dist_forward = np.empty([1,4],dtype=np.float32)
    dist_backward = np.empty([1,4],dtype=np.float32)
    for i in range(4):
        cord = quad_gt[i,:]
        cord_f = quad_gt[(i+1)%4,:]
        cord_b = quad_gt[(i+3)%4,:]
        dist_forward[0,i] = distL2(cord,cord_f)
        dist_backward[0,i] = distL2(cord,cord_b)
        if dist_forward[0,i] >= dist_backward[0,i]:
            shrink_long[i,:] = shrink_line(cord,cord_f,ratio=ratio,r=dist_backward[0,i])
        else:
            shrink_long[i,:] = shrink_line(cord,cord_b,ratio=ratio,r=dist_forward[0,i])
    for i in range(4):
        cord = shrink_long[i, :]
        cord_f = shrink_long[(i + 1) % 4, :]
        cord_b = shrink_long[(i + 3) % 4, :]
        if dist_forward[0,i] >= dist_backward[0,i]:
            shrink_short[i,:] = shrink_line(cord,cord_b,ratio=ratio,r=dist_backward[0,i])
        else:
            shrink_short[i,:] = shrink_line(cord,cord_f,ratio=ratio,r=dist_forward[0,i])
    shrink = np.reshape(shrink_short,(1,8))
    return shrink,shrink_short

def p_to_l_dist(l1,l2,p):
    '''calculate the distance between a point p and a line l1l2'''
    #calculate k,b
    if l2[0] == l1[0]:
        distance = p[0] - l1[0]
    else:
        if l2[1] == l1[1]:
            distance = p[1] - l1[1]
        else:
            k = (l2[1] - l1[1]) / (l2[0] - l1[0])
            b = l1[1] - k * l1[0]
            distance = abs(k * p[0] - p[1] + b) / ((-1) * (-1) + k * k) ** 0.5
    return abs(distance)