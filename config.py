#!/usr/bin/env python

"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Author: Varun Jampani
"""

import os

CAFFEDIR = './lib/video_prop_networks/lib/caffe/'
DATADIR = './data/BSR/BSDS500/data/'
LISTDIR = './data/'

IMG_LIST = {}
IMG_FOLDER = {}
GT_FOLDER = {}

IMG_LIST['TRAIN'] = LISTDIR + 'train.txt'
IMG_FOLDER['TRAIN'] = DATADIR + 'images/train/'
GT_FOLDER['TRAIN'] = DATADIR + '/groundTruth/train/'

IMG_LIST['VAL'] = LISTDIR + 'val.txt'
IMG_FOLDER['VAL'] = DATADIR + 'images/val/'
GT_FOLDER['VAL'] = DATADIR + '/groundTruth/val/'

IMG_LIST['TEST'] = LISTDIR + 'test.txt'
IMG_FOLDER['TEST'] = DATADIR + 'images/test/'
GT_FOLDER['TEST'] = DATADIR + '/groundTruth/test/'

IGNORE_VALUE = 255
RAND_SEED = 2356

TRAIN_BATCH_SIZE = 6
TRAIN_PATCH_WIDTH = 201
TRAIN_PATCH_HEIGHT = 201
