/*
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Author: Varun Jampani
*/

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/rel_to_abs_index_layer.hpp"


namespace caffe {

/*
Setup function
*/
template <typename Dtype>
void RelToAbsIndexLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  RelToAbsIndexParameter rel_to_abs_index_param = this->layer_param_.rel_to_abs_index_param();

  num_spixels_h_ = rel_to_abs_index_param.num_spixels_h();
  num_spixels_w_ = rel_to_abs_index_param.num_spixels_w();
  num_spixels_ = num_spixels_h_ * num_spixels_w_;

  CHECK_EQ(bottom[1]->num(), num_)
    << "Blob dim-0 (num) should be same for bottom blobs.";

  CHECK_EQ(bottom[0]->channels(), 1)
    << "Relative index blob has more than one channel.";

  CHECK_EQ(bottom[1]->channels(), 1)
    << "Superpixl init blob has more than one channel.";

  top[0]->Reshape(num_, 1, height_, width_);
}

template <typename Dtype>
void RelToAbsIndexLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
    top[0]->Reshape(num_, 1, height_, width_);
}

/*
Forward CPU function
*/
template <typename Dtype>
void RelToAbsIndexLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  caffe_set(top[0]->count(), (Dtype)0., top[0]->mutable_cpu_data());

  Dtype* top_data = top[0]->mutable_cpu_data();

  for (unsigned int n = 0; n < num_; ++n) {
    for (unsigned int y = 0; y < height_; ++y) {
      for (unsigned int x = 0; x < width_; ++x) {

        const int r_idx = int(bottom[0]->data_at(n, 0, y, x));
        const int r_idx_h = r_idx / 3 - 1;
        int r_idx_w = r_idx % 3 - 1;

        const int init_spix_index = int(bottom[1]->data_at(n, 0, y, x));
        int spix_idx_h = init_spix_index + r_idx_h * num_spixels_w_;
        if (spix_idx_h >= num_spixels_ || spix_idx_h <= -1) {
          spix_idx_h = init_spix_index;
        }

        if (((spix_idx_h + 1) % num_spixels_w_) == 0 && r_idx_w == 1) {
          r_idx_w = 0;
        } else if ((spix_idx_h % num_spixels_w_) == 0 && r_idx_w == -1) {
          r_idx_w = 0;
        }

        int spix_idx_w = spix_idx_h + r_idx_w;
        if (spix_idx_w < num_spixels_ && spix_idx_w > -1) {
          top_data[top[0]->offset(n, 0, y, x)] = float(spix_idx_w);
        } else {
          top_data[top[0]->offset(n, 0, y, x)] = float(spix_idx_h);
        }

        // int r_label = int(bottom[0]->data_at(n, 0, y, x));
        // int r_label_h = r_label / 3 - 1;
        // int r_label_w = r_label % 3 - 1;
        // int init_spix_index = int(bottom[1]->data_at(n, 0, y, x));
        // int spix_idx_h = init_spix_index + r_label_h * num_spixels_w_;
        // int spix_idx_w = init_spix_index;
        //
        // if (spix_idx_h < num_spixels_ && spix_idx_h > -1) {
        //   spix_idx_w = spix_idx_h + r_label_w;
        // }
        // if (spix_idx_w < num_spixels_ && spix_idx_w > -1) {
        //   top_data[top[0]->offset(n, 0, y, x)] = float(spix_idx_w);
        // }
      }
    }
  }
}


/*
Backward CPU function (NOT_IMPLEMENTED for now)
 */
template <typename Dtype>
void RelToAbsIndexLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(RelToAbsIndexLayer);
#endif

INSTANTIATE_CLASS(RelToAbsIndexLayer);
REGISTER_LAYER_CLASS(RelToAbsIndex);

}  // namespace caffe
