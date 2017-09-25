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
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import cv2

#standard image height and width
stand_img_h = 512
stand_img_w = 512

def display_image(image, gray):
    dis_image = image.astype(np.uint8)
    plt.figure()

    if gray:
        plt.imshow(dis_image, cmap='gray')
    else:
        plt.imshow(dis_image)

def display_polygons(polys,color):
    '''
    :param polys: [N,4,2]
    :param color:
    :return:
    '''
    for p in range(polys.shape[0]):
        current_axis = plt.gca()
        current_axis.add_patch(
        Polygon(xy=polys[p, :, :], linewidth=1, alpha=1, fill=None, edgecolor=color))

def canny_generation(img, G_ksize):
    '''
    generate a weight map based on canny
    :param img: single-channel gray image
    :param G_ksize: Gaussian ksize used in GaussianBlur (must be odd)
    :param bias:
    :return:
    '''
    canny = cv2.Canny(img, 2, 50)
    canny = np.array(canny, dtype=np.float32)

    canny_blur = cv2.GaussianBlur(canny, (G_ksize, G_ksize), 0)
    print(canny_blur.shape, canny_blur.dtype)

    # from demo.evaluate_once import display_image
    # display_image(canny_blur, gray='False')

    canny_max = canny_blur.max()
    regular = canny_max /2

    canny_weight = canny_blur / regular
    print(regular)
    print(canny_weight)
    # display_image(canny_weight, gray='False')

    return canny, canny_blur, canny_weight

def create_random_sample(image,quad):
    '''
    :param image: [h,w,3]
    :param quad: a list of quad
    :return: a cropped image and a quad array
    '''
    random_scale = np.array([0.5, 1, 2.0, 3.0])
    rd_scale = np.random.choice(random_scale)
    print(rd_scale)
    image = cv2.resize(image, dsize=None, fx=rd_scale, fy=rd_scale)
    quad = np.array(quad)
    quad *= rd_scale
    return image,quad

def crop_area(im, polys, crop_background=False, max_tries=50):
    '''
    make random crop from the input image
    :param im:
    :param polys:
    :param tags:
    :param crop_background:
    :param max_tries:
    :return:
    '''
    h, w, _ = im.shape
    pad_h = h//10
    pad_w = w//10
    h_array = np.zeros((h + pad_h*2), dtype=np.int32)
    w_array = np.zeros((w + pad_w*2), dtype=np.int32)
    polys = np.array(polys)
    polys = np.reshape(polys,[-1,4,2])
    for poly in polys:
        poly = np.round(poly, decimals=0).astype(np.int32)
        minx = np.min(poly[:, 0])
        maxx = np.max(poly[:, 0])
        w_array[minx+pad_w:maxx+pad_w] = 1
        miny = np.min(poly[:, 1])
        maxy = np.max(poly[:, 1])
        h_array[miny+pad_h:maxy+pad_h] = 1
    # ensure the cropped area not across a text
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]
    if len(h_axis) == 0 or len(w_axis) == 0:
        return im, polys
    for i in range(max_tries):
        xx = np.random.choice(w_axis, size=2)
        xmin = np.min(xx) - pad_w
        xmax = np.max(xx) - pad_w
        xmin = np.clip(xmin, 0, w-1)
        xmax = np.clip(xmax, 0, w-1)
        yy = np.random.choice(h_axis, size=2)
        ymin = np.min(yy) - pad_h
        ymax = np.max(yy) - pad_h
        ymin = np.clip(ymin, 0, h-1)
        ymax = np.clip(ymax, 0, h-1)
        if xmax - xmin < 5 or ymax - ymin < 5:
            # area too small
            continue

        if len(polys) !=0:
            poly_axis_in_area = (polys[:, :, 0] >= xmin) & (polys[:, :, 0] <= xmax) \
                                & (polys[:, :, 1] >= ymin) & (polys[:, :, 1] <= ymax)
            selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
        else:
            selected_polys = []
        if len(selected_polys) == 0:
            # no text in this area
            if crop_background:
                return im[ymin:ymax+1, xmin:xmax+1, :], polys[selected_polys]
            else:
                continue
        im = im[ymin:ymax+1, xmin:xmax+1, :]
        polys = polys[selected_polys]
        polys[:, :, 0] -= xmin
        polys[:, :, 1] -= ymin
        return im, polys

    return im, polys

def pad_and_resize(image, quad, input_size = stand_img_h): #quad [N 4 2]
    new_h, new_w, _ = image.shape
    max_h_w_i = np.max([new_h, new_w, input_size])
    im_padded = np.zeros((max_h_w_i, max_h_w_i, 3),dtype=np.uint8)

    rand_xmin = 0
    rand_ymin = 0
    if new_h < max_h_w_i:
        rand_ymin = np.random.randint(0, max_h_w_i - new_h)
    if new_w < max_h_w_i:
        rand_xmin = np.random.randint(0,max_h_w_i-new_w)

    im_padded[(rand_ymin):(rand_ymin + new_h), (rand_xmin):(rand_xmin + new_w), :] = image.copy()
    # im_padded[:new_h, :new_w, :] = image.copy()
    image = im_padded

    new_h, new_w, _ =image.shape
    image = cv2.resize(image, dsize=(input_size, input_size))

    quad[:,:,0] += rand_xmin
    quad[:,:,1] += rand_ymin
    resize_ratio_x = input_size / float(new_w)
    resize_ratio_y = input_size / float(new_h)
    quad[:,:,0] *= resize_ratio_x
    quad[:,:,1] *= resize_ratio_y

    print('random:', rand_xmin, rand_ymin)
    return image,quad