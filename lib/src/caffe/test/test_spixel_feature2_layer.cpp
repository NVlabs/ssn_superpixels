/*
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Author: Varun Jampani
*/

#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/spixel_feature2_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class SpixelFeature2LayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  SpixelFeature2LayerTest()
      : blob_bottom_0_(new Blob<Dtype>(1, 2, 4, 3)),
        blob_bottom_1_(new Blob<Dtype>(1, 9, 4, 3)),
        blob_bottom_2_(new Blob<Dtype>(1, 1, 4, 3)),
        blob_top_(new Blob<Dtype>()) {
    blob_bottom_vec_.push_back(blob_bottom_0_);
    blob_bottom_vec_.push_back(blob_bottom_1_);
    blob_bottom_vec_.push_back(blob_bottom_2_);
    blob_top_vec_.push_back(blob_top_);

    for (int i = 0; i < blob_bottom_0_->count(); i++){
      blob_bottom_0_->mutable_cpu_data()[i] = i * 10;
    }

    for (int i = 0; i < blob_bottom_1_->count(); i++){
      blob_bottom_1_->mutable_cpu_data()[i] = 0.1;
    }

    blob_bottom_2_->mutable_cpu_data()[0] = 1;
    blob_bottom_2_->mutable_cpu_data()[1] = 0;
    blob_bottom_2_->mutable_cpu_data()[2] = 2;
    blob_bottom_2_->mutable_cpu_data()[3] = 1;
    blob_bottom_2_->mutable_cpu_data()[4] = 0;
    blob_bottom_2_->mutable_cpu_data()[5] = 0;
    blob_bottom_2_->mutable_cpu_data()[6] = 3;
    blob_bottom_2_->mutable_cpu_data()[7] = 3;
    blob_bottom_2_->mutable_cpu_data()[8] = 3;
    blob_bottom_2_->mutable_cpu_data()[9] = 0;
    blob_bottom_2_->mutable_cpu_data()[10] = 1;
    blob_bottom_2_->mutable_cpu_data()[11] = 1;

  }
  virtual ~SpixelFeature2LayerTest() {
    delete blob_bottom_0_;
    delete blob_bottom_1_;
    delete blob_bottom_2_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SpixelFeature2LayerTest, TestDtypesAndDevices);

TYPED_TEST(SpixelFeature2LayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SpixelFeatureTwoParameter* spixel_param =
    layer_param.mutable_spixel_feature2_param();
  spixel_param->set_num_spixels_h(2);
  spixel_param->set_num_spixels_w(2);
  SpixelFeature2Layer<Dtype> layer(layer_param);

  GradientChecker<Dtype> checker(0.01, 0.01);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 1);
}

}  // namespace caffe
