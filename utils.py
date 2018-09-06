#!/usr/bin/env python

"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Author: Varun Jampani
"""

import numpy as np
from init_caffe import *
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy import interpolate
from skimage.segmentation import mark_boundaries

global g_rel_label
global g_spix_index_init
global g_new_spix_index


def get_rand_scale_factor():

    rand_factor = np.random.normal(1, 0.75)

    s_factor = np.min((3.0, rand_factor))
    s_factor = np.max((0.75, s_factor))

    return s_factor


def initialize_net_weight(net):

    for param_key in net.params.keys():

        # Initialize neighborhood concatenator (convolution layer)
        if param_key.startswith('concat_spixel_feat'):
            num_channels = int(param_key.rsplit('_', 1)[-1])
            for j in range(num_channels):
                for i in range(9):
                    net.params[param_key][0].data[9 * j + i, :, i / 3, i % 3] = 1.0


        # Initialize pixel feature concatenator
        if param_key.startswith('img_concat_pixel_feat'):
            net.params[param_key][0].data[:] = 1.0

        # Initialize spixel feature concatenator
        if param_key == 'repmat_spixel_feat':
            net.params['repmat_spixel_feat'][0].data[:] = 1.0

        # Initialize pixel-spixel distance computation layer
        if param_key.startswith('pixel_spixel_dist_conv'):
            num_channels = int(param_key.rsplit('_', 1)[-1])
            for j in range(9):
                for i in range(num_channels):
                    net.params[param_key][0].data[j, 9 * i + j, 0, 0] = 1.0

        # Initialize scale spixel feature computation layer
        if param_key.startswith('scale_spixel_feat'):
            num_channels = int(param_key.rsplit('_', 1)[-1])
            for j in range(num_channels):
                for i in range(num_channels):
                    net.params['scale_spixel_feat'][0].data[j, 5 * i + j, 0, 0] = 1.0

    return net


def convert_rel_to_spixel_label(rel_label, spix_index,
                                num_spixels_h, num_spixels_w):
    height = rel_label.shape[0]
    width = rel_label.shape[1]
    num_spixels = num_spixels_h * num_spixels_w
    for i in range(height):
        for j in range(width):
            r_label = rel_label[i, j]
            r_label_h = r_label / 3 - 1
            r_label_w = r_label % 3 - 1

            spix_idx_h = spix_index[i, j] + r_label_h * num_spixels_w

            if spix_idx_h < num_spixels and spix_idx_h > -1:
                spix_idx_w = spix_idx_h + r_label_w
            else:
                spix_idx_w = spix_index[i,j]
            if spix_idx_w < num_spixels and spix_idx_w > -1:
                spix_index[i, j] = spix_idx_w

    return spix_index


def visualize_spixels(given_img, spix_index):
    spixel_image = mark_boundaries(given_img / 255., spix_index.astype(int), color = (1,1,1))
    plt.imshow(spixel_image); plt.show();


def get_spixel_image(given_img, spix_index):
    spixel_image = mark_boundaries(given_img / 255., spix_index.astype(int), color = (1,1,1))
    return spixel_image


def get_spixel_init(num_spixels, img_width, img_height):

    k = num_spixels
    k_w = int(np.floor(np.sqrt(k * img_width / img_height)))
    k_h = int(np.floor(np.sqrt(k * img_height / img_width)))

    spixel_height = img_height / (1. * k_h)
    spixel_width = img_width / (1. * k_w)

    h_coords = np.arange(-spixel_height / 2., img_height + spixel_height - 1,
                         spixel_height)
    w_coords = np.arange(-spixel_width / 2., img_width + spixel_width - 1,
                         spixel_width)
    h_grid, w_grid = np.meshgrid(h_coords, w_coords, indexing = 'ij')
    spix_values = np.int32(np.arange(0, k_w * k_h).reshape((k_h, k_w)))
    spix_values = np.pad(spix_values, 1, 'symmetric')
    f = interpolate.RegularGridInterpolator((h_coords, w_coords), spix_values, method='nearest')

    all_h_coords = np.arange(0, img_height, 1)
    all_w_coords = np.arange(0, img_width, 1)
    all_grid = np.array(np.meshgrid(all_h_coords, all_w_coords, indexing = 'ij'))
    all_points = np.reshape(all_grid, (2, img_width * img_height)).transpose()

    spixel_initmap = f(all_points).reshape((img_height,img_width))

    feat_spixel_initmap = spixel_initmap
    return [spixel_initmap, feat_spixel_initmap, k_w, k_h]
