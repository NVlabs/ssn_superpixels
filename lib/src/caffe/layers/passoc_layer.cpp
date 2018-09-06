/*
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Author: Varun Jampani
*/

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/passoc_layer.hpp"

#include <cmath>


namespace caffe {

  // This layer computes pairwise eucledian squared distance between
  // pixel (bottom-0) and surrounding superpixels (bottom-1).
  //
  // bottom[0] is of size NxCxHxW - bottom data
  // bottom[1] is of size NxCx1xK - spixel data
  // bottom[2] is of size Nx1xHxW - spixel index
  // top[0] is of size Nx9xHxW

/*
Setup function
*/
template <typename Dtype>
void PassocLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();

  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  SpixelFeatureTwoParameter spixel_param = this->layer_param_.spixel_feature2_param();

  if (spixel_param.has_scale_value()) {
    scale_value_ = this->layer_param_.spixel_feature2_param().scale_value();
  } else {
    scale_value_ = 1.0;
  }

  if (spixel_param.has_num_spixels_h()) {
    num_spixels_h_ = this->layer_param_.spixel_feature2_param().num_spixels_h();
  } else {
    LOG(FATAL) << "Undefined vertical number of superpixels (num_spixels_h)";
  }

  if (spixel_param.has_num_spixels_w()) {
    num_spixels_w_ = this->layer_param_.spixel_feature2_param().num_spixels_w();
  } else {
    LOG(FATAL) << "Undefined horizontal number of superpixels (num_spixels_w)";
  }

  num_spixels_ = num_spixels_h_ * num_spixels_w_;

  CHECK_EQ(bottom[1]->num(), num_)
    << "dim-0 (num) should be same for bottom blobs.";

  CHECK_EQ(bottom[1]->channels(), channels_)
    << "dim-1 (channels) should be same for bottom blobs.";

  CHECK_EQ(bottom[1]->width(), num_spixels_)
    << "bottom-1 size do not match with given number of superpixels.";

  CHECK_EQ(bottom[2]->num(), num_)
    << "Blob dim-2 (num) should be same for bottom blobs.";

  CHECK_EQ(bottom[2]->channels(), 1)
    << "Spixel index blob has more than one channel.";

  top[0]->Reshape(num_, 9, height_, width_);
}

template <typename Dtype>
void PassocLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(num_, 9, height_, width_);
}

// ((n * channels() + c) * height() + h) * width() + w;

/*
Forward CPU function
*/
template <typename Dtype>
void PassocLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

/*
Backward CPU function
 */
template <typename Dtype>
void PassocLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(PassocLayer);
#endif

INSTANTIATE_CLASS(PassocLayer);
REGISTER_LAYER_CLASS(Passoc);

}  // namespace caffe
