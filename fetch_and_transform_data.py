#!/usr/bin/env python

"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Author: Varun Jampani
"""

import numpy as np
import scipy
from scipy.misc import fromimage
from scipy.io import loadmat
from skimage.color import rgb2lab
from skimage.util import img_as_float
from skimage import io

from utils import *
from config import *
from init_caffe import *

from random import Random
myrandom = Random(RAND_SEED)

def transform_and_get_image(im, max_spixels, out_size):

    height = im.shape[0]
    width = im.shape[1]

    out_height = out_size[0]
    out_width = out_size[1]

    pad_height = out_height - height
    pad_width = out_width - width
    im = np.lib.pad(im, ((0, pad_height), (0, pad_width), (0, 0)), 'constant',
                    constant_values=-10)

    transformer = caffe.io.Transformer({'img': (1, 3, out_size[0],
                                                out_size[1])})
    transformer.set_transpose('img', (2, 0, 1))

    im = np.asarray(transformer.preprocess('img', im))
    im = np.expand_dims(im, axis=0)

    return im

def transform_and_get_spixel_init(max_spixels, out_size):

    out_height = out_size[0]
    out_width = out_size[1]

    spixel_init, feat_spixel_initmap, k_w, k_h = \
        get_spixel_init(max_spixels, out_width, out_height)
    spixel_init = spixel_init[None, None, :, :]
    feat_spixel_initmap = feat_spixel_initmap[None, None, :, :]

    return spixel_init, feat_spixel_initmap, k_h, k_w


def convert_label(label):

    problabel = np.zeros((1, 50, label.shape[0], label.shape[1])).astype(np.float32)

    ct = 0
    for t in np.unique(label).tolist():
        if ct >= 50:
            print(np.unique(label).shape)
            break
        else:
            problabel[:, ct, :, :] = (label == t)
        ct = ct + 1

    label2 = np.squeeze(np.argmax(problabel, axis = 1))

    return label2, problabel

def fetch_and_transform_data(imgname,
                             data_type,
                             out_types,
                             max_spixels):

    image_folder = IMG_FOLDER[data_type]
    image_filename = image_folder + imgname + '.jpg'
    image = img_as_float(io.imread(image_filename))
    im = rgb2lab(image)

    gt_folder = GT_FOLDER[data_type]
    gt_filename = gt_folder + imgname + '.mat'
    gtseg_all = loadmat(gt_filename)
    t = np.random.randint(0, len(gtseg_all['groundTruth'][0]))
    gtseg = gtseg_all['groundTruth'][0][t][0][0][0]
    label, problabel = convert_label(gtseg)

    height = im.shape[0]
    width = im.shape[1]

    out_height = height
    out_width = width

    out_img = transform_and_get_image(im, max_spixels, [out_height, out_width])

    inputs = {}
    for in_name in out_types:
        if in_name == 'img':
            inputs['img'] = out_img
        if in_name == 'spixel_init':
            out_spixel_init, feat_spixel_init, spixels_h, spixels_w = \
                transform_and_get_spixel_init(max_spixels, [out_height, out_width])
            inputs['spixel_init'] = out_spixel_init
        if in_name == 'feat_spixel_init':
            inputs['feat_spixel_init'] = feat_spixel_init
        if in_name == 'label':
            label = np.expand_dims(np.expand_dims(label, axis=0), axis=0)
            inputs['label'] = label
        if in_name == 'problabel':
            inputs['problabel'] = problabel

    return [inputs, height, width]


def scale_image(im, s_factor):

    s_img = scipy.ndimage.zoom(im, (s_factor, s_factor, 1), order = 1)

    return s_img

def scale_label(label, s_factor):

    s_label = scipy.ndimage.zoom(label, (s_factor, s_factor), order = 0)

    return s_label

def fetch_and_transform_patch_data(imgname,
                                   data_type,
                                   out_types,
                                   max_spixels,
                                   patch_size = None):

    s_factor = get_rand_scale_factor()

    image_folder = IMG_FOLDER[data_type]
    image_filename = image_folder + imgname + '.jpg'
    image = img_as_float(io.imread(image_filename))
    image = scale_image(image, s_factor)
    im = rgb2lab(image)

    gt_folder = GT_FOLDER[data_type]
    gt_filename = gt_folder + imgname + '.mat'
    gtseg_all = loadmat(gt_filename)
    t = np.random.randint(0, len(gtseg_all['groundTruth'][0]))
    gtseg = gtseg_all['groundTruth'][0][t][0][0][0]
    gtseg = scale_label(gtseg, s_factor)

    if np.random.uniform(0, 1) > 0.5:
        im = im[:, ::-1, ...]
        gtseg = gtseg[:, ::-1]

    height = im.shape[0]
    width = im.shape[1]

    if patch_size == None:
        out_height = height
        out_width = width
    else:
        out_height = patch_size[0]
        out_width = patch_size[1]

    if out_height > height:
        raise "Patch size is greater than image size"

    if out_width > width:
        raise "Patch size is greater than image size"

    start_row = myrandom.randint(0, height - out_height)
    start_col = myrandom.randint(0, width - out_width)

    im_cropped = im[start_row : start_row + out_height,
                    start_col : start_col + out_width, :]

    out_img = transform_and_get_image(im_cropped, max_spixels, [out_height, out_width])

    gtseg_cropped = gtseg[start_row : start_row + out_height,
                          start_col : start_col + out_width]
    label_cropped, problabel_cropped = convert_label(gtseg_cropped)

    inputs = {}
    for in_name in out_types:
        if in_name == 'img':
            inputs['img'] = out_img
        if in_name == 'spixel_init':
            out_spixel_init, feat_spixel_init, spixels_h, spixels_w = \
                transform_and_get_spixel_init(max_spixels, [out_height, out_width])
            inputs['spixel_init'] = out_spixel_init
        if in_name == 'feat_spixel_init':
            inputs['feat_spixel_init'] = feat_spixel_init
        if in_name == 'label':
            label_cropped = np.expand_dims(np.expand_dims(label_cropped, axis=0), axis=0)
            inputs['label'] = label_cropped
        if in_name == 'problabel':
            inputs['problabel'] = problabel_cropped

    return [inputs, height, width]
