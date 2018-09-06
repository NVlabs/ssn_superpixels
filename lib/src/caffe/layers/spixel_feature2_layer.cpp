/*
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Author: Varun Jampani
*/

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/spixel_feature2_layer.hpp"


namespace caffe {

/*
Setup function
*/
template <typename Dtype>
void SpixelFeature2Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  num_ = bottom[0]->num();
  in_channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  SpixelFeatureTwoParameter spixel_param = this->layer_param_.spixel_feature2_param();

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
    << "Blob dim-0 (num) should be same for bottom blobs.";

  CHECK_EQ(bottom[2]->num(), num_)
    << "Blob dim-0 (num) should be same for bottom blobs.";

  CHECK_EQ(bottom[2]->channels(), 1)
    << "Spixel index blob has more than one channel.";

  CHECK_EQ(bottom[1]->channels(), 9)
    << "Pixel-Spixel association blob should have 9 channels.";

  top[0]->Reshape(num_, in_channels_, 1, num_spixels_);
  spixel_weights_.Reshape(num_, 1, 1, num_spixels_);
}

template <typename Dtype>
void SpixelFeature2Layer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
    top[0]->Reshape(num_, in_channels_, 1, num_spixels_);
}

/*
Forward CPU function
*/
template <typename Dtype>
void SpixelFeature2Layer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    // NOT_IMPLEMENTED;
}

/*
Backward CPU function
 */
template <typename Dtype>
void SpixelFeature2Layer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    // NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(SpixelFeature2Layer);
#endif

INSTANTIATE_CLASS(SpixelFeature2Layer);
REGISTER_LAYER_CLASS(SpixelFeature2);

}  // namespace caffe
