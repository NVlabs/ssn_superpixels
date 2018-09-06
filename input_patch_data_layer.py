#!/usr/bin/env python

"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Author: Varun Jampani
"""

# Input data layer for training
# Adapted from
# https://github.com/LisaAnne/lisa-caffe-public/blob/lstm_video_deploy/examples/LRCN_activity_recognition/sequence_input_layer.py

import io
import numpy as np
import random
from multiprocessing import Pool
from threading import Thread

from init_caffe import *
from config import *
from fetch_and_transform_data import fetch_and_transform_patch_data

from random import Random
myrandom = Random(RAND_SEED)

class DataProcessor(object):
    def __init__(self, patch_size, data_type, top_names, num_spixels):
        self.top_names = top_names
        self.patch_size = patch_size
        self.data_type = data_type
        self.num_spixels = num_spixels

    def __call__(self, imgname):
        data = fetch_and_transform_patch_data(imgname[0],
                                              self.data_type,
                                              self.top_names,
                                              self.num_spixels,
                                              self.patch_size)

        return data


class sequenceGenerator(object):
    def __init__(self, batch_size, data_type,
                 is_random, reset_count):
        self.batch_size = batch_size
        self.image_list = IMG_LIST[data_type]
        self.data_type = data_type
        self.is_random = is_random
        self.idx = 0
        self.rounds = 0
        self.reset_count = reset_count

        f = open(self.image_list, 'r')
        self.img_names = f.readlines()
        f.close()

        self.num_images = len(self.img_names)
        self.rand_generator = Random(RAND_SEED)

    def __call__(self):

        imgname_list = []

        if self.is_random:
            idx_list = self.rand_generator.sample(range(0, self.num_images),
                                                  self.batch_size)
        else:
            idx_list = range(self.idx, self.idx + self.batch_size)
            idx_list = [f % self.num_images for f in idx_list]

        for i in idx_list:
            im_name = self.img_names[i][:-1]
            imgname_list.append(im_name)

        im_info = zip(imgname_list)

        self.idx += self.batch_size
        if self.idx >= self.num_images:
            self.idx = self.idx - self.num_images

        self.rounds += 1
        if self.rounds >= self.reset_count:
            self.rounds = 0
            self.idx = 0
            self.rand_generator = Random(RAND_SEED)

        return im_info


def advance_batch(result, sequence_generator, data_processor, pool):
    im_info = sequence_generator()
    tmp = data_processor(im_info[0])
    result['data'] = pool.map(data_processor, im_info)

class BatchAdvancer():
    def __init__(self, result, sequence_generator, image_processor, pool):
        self.result = result
        self.image_processor = image_processor
        self.sequence_generator = sequence_generator
        self.pool = pool


    def __call__(self):
        return advance_batch(self.result,
                             self.sequence_generator,
                             self.image_processor,
                             self.pool)

class InputRead(caffe.Layer):

    def initialize(self):
        self.is_random_image_order = True
        self.height = TRAIN_PATCH_HEIGHT
        self.width = TRAIN_PATCH_WIDTH

        self.batch_size = TRAIN_BATCH_SIZE
        self.patch_size = [self.height, self.width]
        self.num_tops = 5
        self.top_names = ['img', 'spixel_init', 'feat_spixel_init', 'label', 'problabel']
        self.top_channels = [3, 1, 1, 1, 50]
        self.pool_size = 10

    def setup(self, bottom, top):

        random.seed(RAND_SEED)

        params = self.param_str.split('_')

        if len(params) < 2:
            params = ['TRAIN', '1000000', '100']
            print("Using standard initialization of params:", params)

        data_type = str(params[0])
        reset_count = int(params[1])
        num_spixels = int(params[2])

        self.initialize()

        self.thread_result = {}
        self.thread = None
        pool_size = self.pool_size

        spatial_size = [self.height, self.width]
        self.data_processor = DataProcessor(self.patch_size, data_type,
                                            self.top_names, num_spixels)
        self.sequence_generator = sequenceGenerator(self.batch_size,
                                                    data_type,
                                                    self.is_random_image_order,
                                                    reset_count)

        self.pool = Pool(processes=pool_size)
        self.batch_advancer = BatchAdvancer(self.thread_result,
                                            self.sequence_generator,
                                            self.data_processor,
                                            self.pool)

        self.dispatch_worker()
        print 'Outputs:', self.top_names
        if len(top) != len(self.top_names):
            raise Exception('Incorrect number of outputs (expected %d, got %d)' %
                            (len(self.top_names), len(top)))
        self.join_worker()

    def reshape(self, bottom, top):
        for top_index, name in enumerate(self.top_names):
            shape = (self.batch_size, self.top_channels[top_index],
                    self.height, self.width)
            top[top_index].reshape(*shape)
        pass


    def forward(self, bottom, top):

        if self.thread is not None:
            self.join_worker()

        new_result = {}

        for t, name in enumerate(self.top_names):
            new_result[self.top_names[t]] =\
                [None]*len(self.thread_result['data'][0][0][self.top_names[t]])

        for i in range(self.batch_size):
            for t, name in enumerate(self.top_names):
                top[t].data[i, ...] =\
                    self.thread_result['data'][i][0][self.top_names[t]]

        self.dispatch_worker()


    def dispatch_worker(self):
        assert self.thread is None
        self.thread = Thread(target=self.batch_advancer)
        self.thread.start()


    def join_worker(self):
        assert self.thread is not None
        self.thread.join()
        self.thread = None


    def backward(self, top, propagate_down, bottom):
        pass
