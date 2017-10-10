# -*- coding: utf-8 -*-
# @Time     : 2017/9/27  上午11:59
# @Author   : Zhuxinyue_Sharon
# @Email    : zxybuptsee@163.com
# @File     : nms.py
# @Software : PyCharm

import numpy as np
from shapely.geometry import Polygon as Poly


def intersection(g, p):
    g = Poly(g[:8].reshape((4, 2)))
    p = Poly(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Poly(g).intersection(Poly(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter/union


def weighted_merge(g, p):
    g[:8] = (g[8] * g[:8] + p[8] * p[:8])/(g[8] + p[8])
    g[8] = (g[8] + p[8])
    return g


def standard_nms(S, thres):
    order = np.argsort(S[:, 8])[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = np.array([intersection(S[i], S[t]) for t in order[1:]])

        inds = np.where(ovr <= thres)[0]
        order = order[inds+1]

    return S[keep]


def nms_locality(polys, thres=0.3):
    '''
    locality aware nms of EAST
    :param polys: a N*9 numpy array. first 8 coordinates, then prob
    :return: boxes after nms
    '''
    S = []
    p = None
    for g in polys:
        if p is not None and intersection(g, p) > thres:
            p = weighted_merge(g, p)
        else:
            if p is not None:
                S.append(p)
            p = g
    if p is not None:
        S.append(p)

    if len(S) == 0:
        return np.array([])
    print('Done locality')
    return standard_nms(np.array(polys), thres)

#
# if __name__ == '__main__':
#     # 343,350,448,135,474,143,369,359
#     print(Polygon(np.array([[343, 350], [448, 135],
#                             [474, 143], [369, 359]])).area)
