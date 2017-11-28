# -*- coding: utf-8 -*-
# @Time     : 2017/11/28  下午1:41
# @Author   : Zhuxinyue_Sharon
# @Email    : zxybuptsee@163.com
# @File     : evaluation.py
# @Software : PyCharm

import data.coco.coco_text as coco_text
from shutil import copy
import json

# data_dir_information
root_path = '../data/'
data_src_dir = '/Users/zhuxinyue/Downloads/train2014'
data_dst_dir = root_path + 'text_data_test/'
# img_and_ann_dir = 'img_ann_pair.txt'
# img_and_ann_train = open('img_ann_pair' + '_train.txt','w')
# img_and_ann_valid = open('img_ann_pair' + '_valid.txt','w')
img_and_ann_test = open('img_ann_pair' + '_testnew.txt','w')

# load coco-text data:
cocojson = root_path + 'COCO_Text.json'
ct = coco_text.COCO_Text(cocojson)
print('loading data to the file...')

#get image id
imgIds = ct.getImgIds(imgIds=ct.test)


def load_pairs(img_and_ann,img):
    img_and_ann.write(img['file_name'])
    img_and_ann.write(',')
    img_and_ann.write('img_' + str(img['id']) + '.json')
    img_and_ann.write('\n')


#load text data to a new file text_data
for i in range(len(imgIds)):
    img = ct.loadImgs(imgIds[i])[0]
    if i%100 == 0 :
        print('loding image ',i)
    img_name = data_src_dir + '/' + img['file_name']
    copy(img_name,data_dst_dir)
    annIds = ct.getAnnIds(imgIds=img['id'])
    anns = ct.loadAnns(annIds)

    #write annotation json file
    ann_dir = root_path + 'annotation_json_test/img_' + str(img['id']) + '.json'
    with open(ann_dir, 'w') as json_file:
        json.dump(anns,json_file)

    #write img_ann_pair file, according to the split ratio
    # if i < ratio_train * len(imgIds):
    #     load_pairs(img_and_ann_train,img)
    # else:
    #     if i > (ratio_train + ratio_valid) * len(imgIds):
    #         load_pairs(img_and_ann_test,img)
    #     else:
    #         load_pairs(img_and_ann_valid,img)
    load_pairs(img_and_ann_test, img)

# img_and_ann_train.close()
# img_and_ann_valid.close()
img_and_ann_test.close()