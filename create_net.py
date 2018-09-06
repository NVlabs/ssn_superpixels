#!/usr/bin/env python

"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Author: Varun Jampani
"""

from init_caffe import *
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import tempfile
from loss_functions import *

trans_dim = 15

def normalize(bottom, dim):

    bottom_relu = L.ReLU(bottom)
    sum = L.Convolution(bottom_relu,
                        convolution_param = dict(num_output = 1, kernel_size = 1, stride = 1,
                                                 weight_filler = dict(type = 'constant', value = 1),
                                                 bias_filler = dict(type = 'constant', value = 0)),
                        param=[{'lr_mult':0, 'decay_mult':0}, {'lr_mult':0, 'decay_mult':0}])

    denom = L.Power(sum, power=(-1.0), shift=1e-12)
    denom = L.Tile(denom, axis=1, tiles=dim)

    return L.Eltwise(bottom_relu, denom, operation=P.Eltwise.PROD)

def conv_bn_relu_layer(bottom, num_out):

    conv1 = L.Convolution(bottom,
                          convolution_param = dict(num_output = num_out, kernel_size = 3, stride = 1, pad = 1,
                                                   weight_filler = dict(type = 'gaussian', std = 0.001),
                                                   bias_filler = dict(type = 'constant', value = 0)),
                                                   # engine = P.Convolution.CUDNN),
                          param=[{'lr_mult':1, 'decay_mult':1}, {'lr_mult':2, 'decay_mult':0}])
    bn1 = L.BatchNorm(conv1)
    bn1 = L.ReLU(bn1, in_place = True)

    return bn1

def conv_relu_layer(bottom, num_out):

    conv1 = L.Convolution(bottom,
                          convolution_param = dict(num_output = num_out, kernel_size = 3, stride = 1, pad = 1,
                                                   weight_filler = dict(type = 'gaussian', std = 0.001),
                                                   bias_filler = dict(type = 'constant', value = 0)),
                                                   # engine = P.Convolution.CUDNN),
                          param=[{'lr_mult':1, 'decay_mult':1}, {'lr_mult':2, 'decay_mult':0}])
    conv1 = L.ReLU(conv1, in_place = True)

    return conv1

def cnn_module(bottom, num_out):

    conv1 = conv_bn_relu_layer(bottom, 64)
    conv2 = conv_bn_relu_layer(conv1, 64)
    pool1 = L.Pooling(conv2, pooling_param = dict(kernel_size = 3, stride = 2, pad = 1, pool = P.Pooling.MAX))

    conv3 = conv_bn_relu_layer(pool1, 64)
    conv4 = conv_bn_relu_layer(conv3, 64)
    pool2 = L.Pooling(conv4, pooling_param = dict(kernel_size = 3, stride = 2, pad = 1, pool = P.Pooling.MAX))

    conv5 = conv_bn_relu_layer(pool2, 64)
    conv6 = conv_bn_relu_layer(conv5, 64)

    conv6_upsample = L.Interp(conv6, interp_param = dict(zoom_factor = 4))
    conv6_upsample_crop = L.Crop(conv6_upsample, conv2)

    conv4_upsample = L.Interp(conv4, interp_param = dict(zoom_factor = 2))
    conv4_upsample_crop = L.Crop(conv4_upsample, conv2)

    conv_concat = L.Concat(bottom, conv2, conv4_upsample_crop, conv6_upsample_crop)

    conv7 = conv_relu_layer(conv_concat, num_out)

    conv_comb = L.Concat(bottom, conv7)

    return conv_comb


def compute_assignments(spixel_feat, pixel_features,
                        spixel_init, num_spixels_h,
                        num_spixels_w, num_spixels, num_channels):

    num_channels = int(num_channels)

    pixel_spixel_neg_dist = L.Passoc(pixel_features, spixel_feat, spixel_init,
                                     spixel_feature2_param =\
          dict(num_spixels_h = num_spixels_h, num_spixels_w = num_spixels_w, scale_value = -1.0))

    # Softmax to get pixel-superpixel relative soft-associations
    pixel_spixel_assoc = L.Softmax(pixel_spixel_neg_dist)

    return pixel_spixel_assoc


def compute_final_spixel_labels(pixel_spixel_assoc,
                                spixel_init,
                                num_spixels_h, num_spixels_w):

    # Compute new spixel indices
    rel_label = L.ArgMax(pixel_spixel_assoc, argmax_param = dict(axis = 1),
                         propagate_down = False)
    new_spix_indices = L.RelToAbsIndex(rel_label, spixel_init,
                                       rel_to_abs_index_param = dict(num_spixels_h = int(num_spixels_h),
                                                                     num_spixels_w = int(num_spixels_w)),
                                                                     propagate_down = [False, False])

    return new_spix_indices


def decode_features(pixel_spixel_assoc, spixel_feat, spixel_init,
                    num_spixels_h, num_spixels_w, num_spixels, num_channels):

    num_channels = int(num_channels)

    # Reshape superpixel features to k_h x k_w
    spixel_feat_reshaped = L.Reshape(spixel_feat,
                                      reshape_param = dict(shape = {'dim':[0,0,num_spixels_h,num_spixels_w]}))

    # Concatenate neighboring superixel features
    concat_spixel_feat = L.Convolution(spixel_feat_reshaped,
                                        name = 'concat_spixel_feat_' + str(num_channels),
                                        convolution_param = dict(num_output = num_channels * 9,
                                                                 kernel_size = 3,
                                                                 stride = 1,
                                                                 pad = 1,
                                                                 group = num_channels,
                                                                 bias_term = False),
                                                                 param=[{'name': 'concat_spixel_feat_' + str(num_channels),
                                                                        'lr_mult':0, 'decay_mult':0}])

    # Spread features to pixels
    flat_concat_label = L.Reshape(concat_spixel_feat,
                                  reshape_param = dict(shape = {'dim':[0, 0, 1, num_spixels]}))
    img_concat_spixel_feat = L.Smear(flat_concat_label, spixel_init)

    tiled_assoc = L.Tile(pixel_spixel_assoc,
                         tile_param = dict(tiles = num_channels))

    weighted_spixel_feat = L.Eltwise(img_concat_spixel_feat, tiled_assoc,
                                     eltwise_param = dict(operation = P.Eltwise.PROD))
    recon_feat = L.Convolution(weighted_spixel_feat,
                               name = 'recon_feat_' + str(num_channels),
                               convolution_param = dict(num_output = num_channels,
                                                        kernel_size = 1,
                                                        stride = 1,
                                                        pad = 0,
                                                        group = num_channels,
                                                        bias_term = False,
                                                        weight_filler = dict(type = 'constant', value = 1.0)),
                                                        param=[{'name': 'recon_feat_' + str(num_channels),
                                                               'lr_mult':0, 'decay_mult':0}])

    return recon_feat


def exec_iter(spixel_feat, trans_features, spixel_init,
              num_spixels_h, num_spixels_w, num_spixels,
              trans_dim):

    # Compute pixel-superpixel assignments
    pixel_assoc = \
        compute_assignments(spixel_feat, trans_features,
                            spixel_init, num_spixels_h,
                            num_spixels_w, num_spixels, trans_dim)
    # Compute superpixel features from pixel assignments
    spixel_feat1 = L.SpixelFeature2(trans_features,
                                    pixel_assoc,
                                    spixel_init,
                                    spixel_feature2_param =\
        dict(num_spixels_h = num_spixels_h, num_spixels_w = num_spixels_w))

    return spixel_feat1


def create_ssn_net(img_height, img_width,
                   num_spixels, pos_scale, color_scale,
                   num_spixels_h, num_spixels_w, num_steps,
                   phase = None):

    n = caffe.NetSpec()

    if phase == 'TRAIN':
        n.img, n.spixel_init, n.feat_spixel_init, n.label, n.problabel = \
            L.Python(python_param = dict(module = "input_patch_data_layer", layer = "InputRead", param_str = "TRAIN_1000000_" + str(num_spixels)),
                     include = dict(phase = 0),
                     ntop = 5)
    elif phase == 'TEST':
        n.img, n.spixel_init, n.feat_spixel_init, n.label, n.problabel = \
            L.Python(python_param = dict(module = "input_patch_data_layer", layer = "InputRead", param_str = "VAL_10_" + str(num_spixels)),
                     include = dict(phase = 1),
                     ntop = 5)
    else:
        n.img = L.Input(shape=[dict(dim=[1, 3, img_height, img_width])])
        n.spixel_init = L.Input(shape=[dict(dim=[1, 1, img_height, img_width])])
        n.feat_spixel_init = L.Input(shape=[dict(dim=[1, 1, img_height, img_width])])


    n.pixel_features = L.PixelFeature(n.img,
                                      pixel_feature_param = dict(type = P.PixelFeature.POSITION_AND_RGB,
                                                                 pos_scale = float(pos_scale),
                                                                 color_scale = float(color_scale)))

    ### Transform Pixel features
    n.trans_features = cnn_module(n.pixel_features, trans_dim)


    # Initial Superpixels
    n.init_spixel_feat = L.SpixelFeature(n.trans_features, n.feat_spixel_init,
                                         spixel_feature_param =\
        dict(type = P.SpixelFeature.AVGRGB, rgb_scale = 1.0, ignore_idx_value = -10,
             ignore_feature_value = 255, max_spixels = int(num_spixels)))

    ### Iteration-1
    n.spixel_feat1 = exec_iter(n.init_spixel_feat, n.trans_features,
                               n.spixel_init, num_spixels_h,
                               num_spixels_w, num_spixels, trans_dim)

    ### Iteration-2
    n.spixel_feat2 = exec_iter(n.spixel_feat1, n.trans_features,
                               n.spixel_init, num_spixels_h,
                               num_spixels_w, num_spixels, trans_dim)

    ### Iteration-3
    n.spixel_feat3 = exec_iter(n.spixel_feat2, n.trans_features,
                               n.spixel_init, num_spixels_h,
                               num_spixels_w, num_spixels, trans_dim)

    ### Iteration-4
    n.spixel_feat4 = exec_iter(n.spixel_feat3, n.trans_features,
                               n.spixel_init, num_spixels_h,
                               num_spixels_w, num_spixels, trans_dim)

    if num_steps == 5:
        ### Iteration-5
        n.final_pixel_assoc  = \
            compute_assignments(n.spixel_feat4, n.trans_features,
                                n.spixel_init, num_spixels_h,
                                num_spixels_w, num_spixels, trans_dim)

    elif num_steps == 10:
        ### Iteration-5
        n.spixel_feat5 = exec_iter(n.spixel_feat4, n.trans_features,
                                   n.spixel_init, num_spixels_h,
                                   num_spixels_w, num_spixels, trans_dim)

        ### Iteration-6
        n.spixel_feat6 = exec_iter(n.spixel_feat5, n.trans_features,
                                   n.spixel_init, num_spixels_h,
                                   num_spixels_w, num_spixels, trans_dim)

        ### Iteration-7
        n.spixel_feat7 = exec_iter(n.spixel_feat6, n.trans_features,
                                   n.spixel_init, num_spixels_h,
                                   num_spixels_w, num_spixels, trans_dim)

        ### Iteration-8
        n.spixel_feat8 = exec_iter(n.spixel_feat7, n.trans_features,
                                   n.spixel_init, num_spixels_h,
                                   num_spixels_w, num_spixels, trans_dim)

        ### Iteration-9
        n.spixel_feat9 = exec_iter(n.spixel_feat8, n.trans_features,
                                   n.spixel_init, num_spixels_h,
                                   num_spixels_w, num_spixels, trans_dim)

        ### Iteration-10
        n.final_pixel_assoc  = \
            compute_assignments(n.spixel_feat9, n.trans_features,
                                n.spixel_init, num_spixels_h,
                                num_spixels_w, num_spixels, trans_dim)


    if phase == 'TRAIN' or phase == 'TEST':

        # Compute final spixel features
        n.new_spixel_feat = L.SpixelFeature2(n.pixel_features,
                                             n.final_pixel_assoc,
                                             n.spixel_init,
                                             spixel_feature2_param =\
            dict(num_spixels_h = num_spixels_h, num_spixels_w = num_spixels_w))


        n.new_spix_indices = compute_final_spixel_labels(n.final_pixel_assoc,
                                                         n.spixel_init,
                                                         num_spixels_h, num_spixels_w)
        n.recon_feat2 = L.Smear(n.new_spixel_feat, n.new_spix_indices,
                                propagate_down = [True, False])
        n.loss1, n.loss2 = position_color_loss(n.recon_feat2, n.pixel_features,
                                               pos_weight = 0.00001,
                                               col_weight = 0.0)


        # Convert pixel labels to spixel labels
        n.spixel_label = L.SpixelFeature2(n.problabel,
                                          n.final_pixel_assoc,
                                          n.spixel_init,
                                          spixel_feature2_param =\
            dict(num_spixels_h = num_spixels_h, num_spixels_w = num_spixels_w))
        # Convert spixel labels back to pixel labels
        n.recon_label = decode_features(n.final_pixel_assoc, n.spixel_label, n.spixel_init,
                                        num_spixels_h, num_spixels_w, num_spixels, num_channels = 50)


        n.recon_label = L.ReLU(n.recon_label, in_place = True)
        n.recon_label2 = L.Power(n.recon_label, power_param = dict(shift = 1e-10))
        n.recon_label3 = normalize(n.recon_label2, 50)
        n.loss3 = L.LossWithoutSoftmax(n.recon_label3, n.label,
                                       loss_param = dict(ignore_label = 255),
                                       loss_weight = 1.0)

    else:
        n.new_spix_indices = compute_final_spixel_labels(n.final_pixel_assoc,
                                                         n.spixel_init,
                                                         num_spixels_h, num_spixels_w)

    return n.to_proto()


def load_ssn_net(img_height, img_width,
                 num_spixels, pos_scale, color_scale,
                 num_spixels_h, num_spixels_w, num_steps):

    net_proto = create_ssn_net(img_height, img_width,
                               num_spixels, pos_scale, color_scale,
                               num_spixels_h, num_spixels_w, int(num_steps))

    # Save to temporary file and load
    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write(str(net_proto))
    f.close()
    return caffe.Net(f.name, caffe.TEST)


def get_ssn_net(img_height, img_width,
                num_spixels, pos_scale, color_scale,
                num_spixels_h, num_spixels_w, num_steps,
                phase):

    # Create the prototxt
    net_proto = create_ssn_net(img_height, img_width,
                               num_spixels, pos_scale, color_scale,
                               num_spixels_h, num_spixels_w, int(num_steps), phase)

    # Save to temporary file and load
    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write(str(net_proto))
    f.close()
    return f.name
