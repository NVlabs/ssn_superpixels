/*
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Author: Varun Jampani
*/

#ifndef SPIXEL_FEATURE2_LAYER_HPP_
#define SPIXEL_FEATURE2_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

  template <typename Dtype>
  class SpixelFeature2Layer : public Layer<Dtype> {
   public:
    explicit SpixelFeature2Layer(const LayerParameter& param)
        : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "SpixelFeature2"; }
    virtual inline int ExactNumBottomBlobs() const { return 3; }

   protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    int num_;
    int in_channels_;
    int height_;
    int width_;
    int out_channels_;
    int num_spixels_;
    int num_spixels_h_;
    int num_spixels_w_;

    Blob<Dtype> spixel_weights_;
  };

} //namespace caffe

#endif  // SPIXEL_FEATURE_LAYER_HPP_
