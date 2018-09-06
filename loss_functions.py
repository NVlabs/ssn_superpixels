#!/usr/bin/env python

"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Author: Varun Jampani
"""

from init_caffe import *
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2


def l1_loss(bottom1, bottom2, l_weight):

    diff = L.Eltwise(bottom1, bottom2,
                     eltwise_param = dict(operation = P.Eltwise.SUM, coeff = [1, -1]))
    absval = L.AbsVal(diff)
    loss = L.Reduction(absval,
                       reduction_param = dict(operation = P.Reduction.SUM),
                       loss_weight = l_weight)

    return loss

def centroid_loss2(trans_features, new_spixel_features,
                  new_spix_indices, num_spixels, l_weight):

    centroid_loss = L.EuclideanLoss(trans_features, new_spixel_features, loss_weight = l_weight)

    return centroid_loss

def centroid_pos_color_loss2(trans_features, new_spixel_features,
                 	         num_spixels, l_weight_pos, l_weight_color):

    pos_recon_feat, color_recon_feat = L.Slice(new_spixel_features,
                                               slice_param = dict(axis = 1,
                                                                  slice_point = 2),
                                               ntop = 2)

    pos_pix_feat, color_pix_feat = L.Slice(trans_features,
                                           slice_param = dict(axis = 1,
                                                              slice_point = 2),
                                           ntop = 2)

    pos_loss = L.EuclideanLoss(pos_recon_feat, pos_pix_feat, loss_weight = l_weight_pos)
    color_loss = L.EuclideanLoss(color_recon_feat, color_pix_feat, loss_weight = l_weight_color)

    return pos_loss, color_loss



def centroid_loss(trans_features, computed_spixel_feat,
                  new_spix_indices, num_spixels, l_weight):

    new_spixel_features = L.SpixelFeature(trans_features, new_spix_indices,
                                          spixel_feature_param =\
        dict(type = P.SpixelFeature.AVGRGB, rgb_scale = 1.0, ignore_idx_value = -10,
             ignore_feature_value = 255, max_spixels = int(num_spixels)), propagate_down = [True, False])

    centroid_loss = L.EuclideanLoss(computed_spixel_feat, new_spixel_features, loss_weight = l_weight)

    return centroid_loss

def centroid_pos_color_loss(trans_features, computed_spixel_feat,
                  new_spix_indices, num_spixels, l_weight_pos, l_weight_color):

    new_spixel_features = L.SpixelFeature(trans_features, new_spix_indices,
                                          spixel_feature_param =\
        dict(type = P.SpixelFeature.AVGRGB, rgb_scale = 1.0, ignore_idx_value = -10,
             ignore_feature_value = 255, max_spixels = int(num_spixels)), propagate_down = [True, False])

    pos_recon_feat, color_recon_feat = L.Slice(computed_spixel_feat,
                                               slice_param = dict(axis = 1,
                                                                  slice_point = 2),
                                               ntop = 2)

    pos_pix_feat, color_pix_feat = L.Slice(new_spixel_features,
                                           slice_param = dict(axis = 1,
                                                              slice_point = 2),
                                           ntop = 2)

    pos_loss = L.EuclideanLoss(pos_recon_feat, pos_pix_feat, loss_weight = l_weight_pos)
    color_loss = L.EuclideanLoss(color_recon_feat, color_pix_feat, loss_weight = l_weight_color)

    return pos_loss, color_loss

def crop_x(bottom):
    dummy_data = L.DummyData(dummy_data_param = dict(shape=[dict(dim=[1, 1, 100, 99])]))
    crop_x = L.Crop(bottom, dummy_data,
                    crop_param = dict(offset = [0, 1]))
    return crop_x

def crop_y(bottom):

    dummy_data = L.DummyData(dummy_data_param = dict(shape=[dict(dim=[1, 1, 99, 100])]))
    crop_y = L.Crop(bottom, dummy_data,
                    crop_param = dict(offset = [1, 0]))
    return crop_y


def gradient_x(bottom):
    dummy_data = L.DummyData(dummy_data_param = dict(shape=[dict(dim=[1, 1, 100, 99])]))
    crop_1 = L.Crop(bottom, dummy_data,
                    crop_param = dict(offset = [0, 1]))
    crop_2 = L.Crop(bottom, dummy_data,
                    crop_param = dict(offset = [0, 0]))
    diff = L.Eltwise(crop_1, crop_2,
                     eltwise_param = dict(operation = P.Eltwise.SUM,
                                          coeff = [1.0, -1.0]))
    gradient_x = L.AbsVal(diff)
    return gradient_x

def gradient_y(bottom):

    dummy_data = L.DummyData(dummy_data_param = dict(shape=[dict(dim=[1, 1, 99, 100])]))
    crop_1 = L.Crop(bottom, dummy_data,
                    crop_param = dict(offset = [1, 0]))
    crop_2 = L.Crop(bottom, dummy_data,
                    crop_param = dict(offset = [0, 0]))
    diff = L.Eltwise(crop_1, crop_2,
                     eltwise_param = dict(operation = P.Eltwise.SUM,
                                          coeff = [1.0, -1.0]))
    gradient_y = L.AbsVal(diff)
    return gradient_y

def weight_edges(bottom):
    bottom_avg = L.Convolution(bottom,
                               convolution_param = dict(num_output = 9,
                                                        kernel_size = 1,
                                                        stride = 1,
                                                        pad = 0,
                                                        bias_term = False,
                                                        weight_filler = dict(type = 'constant', value = 1.0)),
                                                        param=[{'lr_mult':0, 'decay_mult':0}])

    weight = L.Exp(bottom_avg, exp_param = dict(scale = -1.0))
    return weight

def weight_edges2(bottom, num_output, power = 1.0):
    bottom_avg = L.Convolution(bottom,
                               convolution_param = dict(num_output = num_output,
                                                        kernel_size = 1,
                                                        stride = 1,
                                                        pad = 0,
                                                        bias_term = False,
                                                        weight_filler = dict(type = 'constant', value = 1.0)),
                                                        param=[{'lr_mult':0, 'decay_mult':0}])

    binarized = L.Power(bottom_avg, power_param = dict(power = power))
    weight = L.Power(binarized, power_param = dict(shift = 1, scale = -1))

    return weight

# To enforce smoothness in pred where there are no edges in img
def smooth_loss(pred, img, l_weight):

    img_x = gradient_x(img)
    img_y = gradient_y(img)

    pred_x = gradient_x(pred)
    pred_y = gradient_y(pred)

    weight_x = weight_edges(img_x)
    weight_y = weight_edges(img_y)

    smoothness_x = L.Eltwise(pred_x, weight_x, operation = P.Eltwise.PROD)
    smoothness_y = L.Eltwise(pred_y, weight_y, operation = P.Eltwise.PROD)

    mean_x_smooth = L.Reduction(smoothness_x,
                                reduction_param = dict(operation = P.Reduction.SUM))
    mean_y_smooth = L.Reduction(smoothness_y,
                                reduction_param = dict(operation = P.Reduction.SUM))

    smooth_loss = L.Eltwise(mean_x_smooth, mean_y_smooth, operation = P.Eltwise.SUM, loss_weight = l_weight)

    return smooth_loss

# Same as smooth_loss but with spixel_init discontinuties taken into account
def smooth_loss3(pred, canny, spixel_init, l_weight):

    spixel_x = gradient_x(spixel_init)
    spixel_y = gradient_y(spixel_init)
    pred_x = gradient_x(pred)
    pred_y = gradient_y(pred)
    weight_init_x = weight_edges2(spixel_x, 9, power = 0.0001)
    weight_init_y = weight_edges2(spixel_y, 9, power = 0.0001)
    w_pred_x = L.Eltwise(pred_x, weight_init_x, operation = P.Eltwise.PROD)
    w_pred_y = L.Eltwise(pred_y, weight_init_y, operation = P.Eltwise.PROD)

    canny_x = crop_x(canny)
    canny_y = crop_y(canny)
    weight_x = weight_edges2(canny_x, 9)
    weight_y = weight_edges2(canny_y, 9)
    smoothness_x = L.Eltwise(w_pred_x, weight_x, operation = P.Eltwise.PROD)
    smoothness_y = L.Eltwise(w_pred_y, weight_y, operation = P.Eltwise.PROD)

    mean_x_smooth = L.Reduction(smoothness_x,
                                reduction_param = dict(operation = P.Reduction.SUM))
    mean_y_smooth = L.Reduction(smoothness_y,
                                reduction_param = dict(operation = P.Reduction.SUM))

    smooth_loss = L.Eltwise(mean_x_smooth, mean_y_smooth, operation = P.Eltwise.SUM, loss_weight = l_weight)

    return smooth_loss

# Same as smooth_loss but with spixel_init discontinuties taken into account
def smooth_loss4(pred, canny, l_weight):

    pred_x = gradient_x(pred)
    pred_y = gradient_y(pred)

    canny_x = crop_x(canny)
    canny_y = crop_y(canny)
    weight_x = weight_edges2(canny_x, 5)
    weight_y = weight_edges2(canny_y, 5)
    smoothness_x = L.Eltwise(pred_x, weight_x, operation = P.Eltwise.PROD)
    smoothness_y = L.Eltwise(pred_y, weight_y, operation = P.Eltwise.PROD)

    mean_x_smooth = L.Reduction(smoothness_x,
                                reduction_param = dict(operation = P.Reduction.SUM))
    mean_y_smooth = L.Reduction(smoothness_y,
                                reduction_param = dict(operation = P.Reduction.SUM))

    smooth_loss = L.Eltwise(mean_x_smooth, mean_y_smooth, operation = P.Eltwise.SUM, loss_weight = l_weight)

    return smooth_loss

# To enfore smoothness in pred
def smooth_loss2(pred, l_weight):

    pred_x = gradient_x(pred)
    pred_y = gradient_y(pred)

    mean_x_smooth = L.Reduction(pred_x,
                                reduction_param = dict(operation = P.Reduction.SUM))
    mean_y_smooth = L.Reduction(pred_y,
                                reduction_param = dict(operation = P.Reduction.SUM))

    smooth_loss = L.Eltwise(mean_x_smooth, mean_y_smooth, operation = P.Eltwise.SUM, loss_weight = l_weight)

    return smooth_loss

def position_color_loss(recon_feat, pixel_features, pos_weight, col_weight):

    pos_recon_feat, color_recon_feat = L.Slice(recon_feat,
                                               slice_param = dict(axis = 1,
                                                                  slice_point = 2),
                                               ntop = 2)

    pos_pix_feat, color_pix_feat = L.Slice(pixel_features,
                                           slice_param = dict(axis = 1,
                                                              slice_point = 2),
                                           ntop = 2)

    pos_loss = L.EuclideanLoss(pos_recon_feat, pos_pix_feat, loss_weight = pos_weight)
    color_loss = L.EuclideanLoss(color_recon_feat, color_pix_feat, loss_weight = col_weight)

    return pos_loss, color_loss
