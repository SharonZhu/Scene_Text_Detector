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
import tensorflow as tf
import numpy as np

import sys
sys.path.append('../')

from nets.vgg16  import VGG16
from nets.fcn import FCN
from EAST import EAST


tf_root_path = 'data/tfrecords/'
tfrecords_filename_train = tf_root_path + 'train_train.tfrecords'
tfrecords_filename_valid = tf_root_path + 'train_valid.tfrecords'
tfrecords_filename_test = tf_root_path + 'train_test.tfrecords'

img_h = 512
img_w = 512
CANNY = False

starter_learing_rate = 1e-3
bound_step = 15000
BATCH_SIZE = 5
CAPACITY = 10 + 3 * BATCH_SIZE
MAX_STEP = 6000
NUM_EPOCH = 1500
min_after_dequeue = 10

SUMMARY_STEP = 500
CKP_STEP = 1000
VAL_STEP = 20000

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'img_raw':tf.FixedLenFeature([],tf.string),
            'ann_raw':tf.FixedLenFeature([],tf.string),
            'img_h': tf.FixedLenFeature([], tf.int64),
            'img_w': tf.FixedLenFeature([], tf.int64)
        }
    )
    # image = tf.decode_raw(features['img_raw'],tf.float64)
    # print(image)
    # image = tf.reshape(image,[512,512,3])


    image = tf.decode_raw(features['img_raw'], tf.float32)
    image = tf.reshape(image,[img_h,img_w,3])
    print(image.shape)

    ann = tf.decode_raw(features['ann_raw'], tf.float32)
    if CANNY:
        ann = tf.reshape(ann, [img_h, img_w, 16])
    else:
        ann = tf.reshape(ann, [img_h, img_w, 15])
    print(ann.shape)

    ##original height and width of an image
    height = features['img_h']
    width = features['img_w']

    return image, ann, height, width



def inputs(set,batch_size,num_epochs):
    if not num_epochs: num_epochs = None
    if set == 'train':
        filename = tfrecords_filename_train
    else:
        if set == 'valid':
            filename = tfrecords_filename_valid
        else:
            filename = tfrecords_filename_test

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [filename], num_epochs=num_epochs)

        # Even when reading in multiple threads, share the filename
        # queue.
        image, ann, height, width= read_and_decode(filename_queue)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        images, sparse_anns, heights, widths= tf.train.shuffle_batch(
            [image, ann, height, width], batch_size=batch_size, num_threads=2,
            capacity=CAPACITY,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=min_after_dequeue)

        return images, sparse_anns, heights, widths


def training(loss):
    '''
    training ops
    :param loss:
    :param learning_rate:
    :return: train_op: the op for training
    '''
    with tf.name_scope('optimizer'):
        #learning rate decay
        global_step = tf.Variable(0,name='global_step',trainable=False)
        boundaries = [bound_step,bound_step*2,bound_step*5]
        values = [starter_learing_rate,starter_learing_rate*0.1,starter_learing_rate*0.01, starter_learing_rate*0.001]
        learning_rate = tf.train.piecewise_constant(global_step,boundaries,values,name='learning_rate')
        print("learning rate = ",learning_rate)
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss,global_step=global_step)
    return train_op



def run_training():
    logs_train_dir = '../outmodel/logs/train'
    logs_val_dir = '../outmodel/logs/val'

    rgb = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, img_h, img_w, 3])
    if CANNY:
        labels = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, img_h, img_w, 16])
    else:
        labels = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, img_h, img_w, 15])

    my_vgg16 = VGG16('../pre_data/vgg16.npy')
    with tf.name_scope('netVgg16') as netVgg16:
        my_vgg16.build(rgb)

    my_fcn = FCN(my_vgg16)
    with tf.name_scope('netFCN') as netFCN:
        '''
        'need_layers = [('pool5', stride, up_fea_num, conv1_ksize, conv2_ksize, conv1_out_feat_num, conv2_out_feat_num), ('pool4', 16, out_fea_num)]'
        '''
        my_fcn.build([('pool5', 2, 64, 1, 3, 128, 128), ('pool4', 2, 64, 1, 3, 96, 64), ('pool3', 2, 32, 1, 3, 64, 32),
                      ('pool2', 4, 32, 1, 3, 32, 32), ('bgr', 3, 32)], fuse_type='concat', debug=False)

    my_east = EAST.EAST(my_fcn.out_layer, task='train', labels=labels)

    with tf.name_scope('netEAST') as netEAST:
        train_loss = my_east.loss(canny=CANNY)

    train_images, train_anns, train_height, train_width = inputs(set='train', batch_size=BATCH_SIZE,num_epochs=NUM_EPOCH)
    print('train_images:',train_images)
    print('train_anns:',train_anns)

    # valid_images, valid_anns, valid_height, valid_width = inputs(set='valid', batch_size=BATCH_SIZE,
    #                                                              num_epochs=NUM_EPOCH)

    train_op = training(train_loss)

    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
        sess.run(init_op)

        saver = tf.train.Saver()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
        val_writer = tf.summary.FileWriter(logs_val_dir, sess.graph)

        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                    break
                #test values of label and logits in a batch:
                tra_images,tra_labels = sess.run([train_images,train_anns])
                has_ans = np.sum(np.split(tra_labels, [1, 14], 3)[0])

                print( 'has_ans', has_ans)
                if has_ans == 0:
                    continue

                print(tra_images.shape,tra_images.dtype)
                print(tra_labels.shape,tra_labels.dtype)
                loss_look = tf.get_collection('loss')
                res = sess.run([train_op, train_loss, loss_look],
                                       feed_dict={rgb: tra_images, labels: tra_labels})

                print('train_loss in step ',step)
                print('tra_loss', res[1])

                print(loss_look)

                for i, x in enumerate(res[2]):
                    print(loss_look[i], x)

                if step % SUMMARY_STEP == 0:
                    pass
                    #print('Step %d, train loss=%.4f'%(step,tra_loss))
                    print(tra_images.shape, tra_labels.shape)
                    summary_str = sess.run(summary_op,feed_dict={rgb: tra_images, labels: tra_labels})
                    train_writer.add_summary(summary_str,step)

                if step % VAL_STEP == 0 or (step+1) == MAX_STEP:
                    val_images,val_labels = sess.run([valid_images,valid_anns])
                    val_loss = sess.run([train_loss],
                                        feed_dict={rgb: val_images, labels: val_labels})
                    print('**val_loss in step ', step)
                    print(val_loss)
                    #print('**  Step %d, val loss = %.4f' % (step, val_loss))
                    summary_str = sess.run(summary_op,feed_dict={rgb: val_images, labels: val_labels})
                    val_writer.add_summary(summary_str, step)

                if step % CKP_STEP == 0 or (step+1) == MAX_STEP:
                    checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

        except tf.errors.OutOfRangeError:
            print('Done training')
        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()

def main(argv=None):
  run_training()

if __name__ == '__main__':
  tf.app.run()
