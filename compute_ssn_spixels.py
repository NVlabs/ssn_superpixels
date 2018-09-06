#!/usr/bin/env python

"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Author: Varun Jampani
"""

import numpy as np
import scipy.io as sio
import os
import scipy
from scipy.misc import fromimage
from scipy.misc import imsave
from PIL import Image
import argparse

from init_caffe import *
from config import *
from utils import *
from fetch_and_transform_data import fetch_and_transform_data, transform_and_get_spixel_init
from create_net import load_ssn_net

import sys
sys.path.append('../lib/cython')
from connectivity import enforce_connectivity

def compute_spixels(data_type, n_spixels, num_steps,
                    caffe_model, out_folder, is_connected = True):

    image_list = IMG_LIST[data_type]

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    p_scale = 0.40
    color_scale = 0.26

    with open(image_list) as list_f:
        for imgname in list_f:
            print(imgname)
            imgname = imgname[:-1]
            [inputs, height, width] = \
                fetch_and_transform_data(imgname, data_type,
                                         ['img', 'label', 'problabel'],
                                         int(n_spixels))

            height = inputs['img'].shape[2]
            width = inputs['img'].shape[3]
            [spixel_initmap, feat_spixel_initmap, num_spixels_h, num_spixels_w] =\
                transform_and_get_spixel_init(int(n_spixels), [height, width])

            dinputs = {}
            dinputs['img'] = inputs['img']
            dinputs['spixel_init'] = spixel_initmap
            dinputs['feat_spixel_init'] = feat_spixel_initmap

            pos_scale_w = (1.0 * num_spixels_w) / (float(p_scale) * width)
            pos_scale_h = (1.0 * num_spixels_h) / (float(p_scale) * height)
            pos_scale = np.max([pos_scale_h, pos_scale_w])

            net = load_ssn_net(height, width, int(num_spixels_w * num_spixels_h),
                               float(pos_scale), float(color_scale),
                               num_spixels_h, num_spixels_w, int(num_steps))

            if caffe_model is not None:
                net.copy_from(caffe_model)
            else:
                net = initialize_net_weight(net)

            num_spixels = int(num_spixels_w * num_spixels_h)
            result = net.forward_all(**dinputs)

            given_img = fromimage(Image.open(IMG_FOLDER[data_type] + imgname + '.jpg'))
            spix_index = np.squeeze(net.blobs['new_spix_indices'].data).astype(int)

            if enforce_connectivity:
                segment_size = (given_img.shape[0] * given_img.shape[1]) / (int(n_spixels) * 1.0)
                min_size = int(0.06 * segment_size)
                max_size = int(3 * segment_size)
                spix_index = enforce_connectivity(spix_index[None, :, :], min_size, max_size)[0]

            spixel_image = get_spixel_image(given_img, spix_index)
    	    out_img_file = out_folder + imgname + '_bdry.jpg'
            imsave(out_img_file, spixel_image)
            out_file = out_folder + imgname + '.npy'
            np.save(out_file, spix_index)

    return


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--datatype', type=str, required=True)
    parser.add_argument('--n_spixels', type=int, required=True)
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--caffemodel', type=str, required=True)
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--is_connected', type=bool, default=True)

    var_args = parser.parse_args()
    compute_spixels(var_args.datatype,
                    var_args.n_spixels,
                    var_args.num_steps,
                    var_args.caffemodel,
                    var_args.result_dir,
                    var_args.is_connected)

if __name__ == '__main__':
    main()
