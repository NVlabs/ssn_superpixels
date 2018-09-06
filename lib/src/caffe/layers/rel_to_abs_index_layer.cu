/*
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Author: Varun Jampani
*/

#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/layers/rel_to_abs_index_layer.hpp"


namespace caffe {

template <typename Dtype>
  __global__ void RelToAbsIndexForwardGPU(const int nthreads,
    const Dtype* rel_label_data, const Dtype* spixel_init_data,
    const int num_spixels, const int num_spixels_w, Dtype* top_data) {
      CUDA_KERNEL_LOOP(index, nthreads) {

        const int r_idx = rel_label_data[index];
        const int r_idx_h = r_idx / 3 - 1;
        int r_idx_w = r_idx % 3 - 1;

        const int init_spix_index = spixel_init_data[index];
        int spix_idx_h = init_spix_index + r_idx_h * num_spixels_w;
        if (spix_idx_h >= num_spixels || spix_idx_h <= -1) {
          spix_idx_h = init_spix_index;
        }

        if (((spix_idx_h + 1) % num_spixels_w) == 0 && r_idx_w == 1) {
          r_idx_w = 0;
        } else if ((spix_idx_h % num_spixels_w) == 0 && r_idx_w == -1) {
          r_idx_w = 0;
        }

        int spix_idx_w = spix_idx_h + r_idx_w;
        if (spix_idx_w < num_spixels && spix_idx_w > -1) {
          top_data[index] = float(spix_idx_w);
        } else {
          top_data[index] = float(spix_idx_h);
        }


        // const int r_label = rel_label_data[index];
        // const int r_label_h = r_label / 3 - 1;
        // const int r_label_w = r_label % 3 - 1;
        //
        // const int init_spix_index = spixel_init_data[index];
        // int spix_idx_h = init_spix_index + r_label_h * num_spixels_w;
        // int spix_idx_w = init_spix_index;
        //
        // if (spix_idx_h < num_spixels && spix_idx_h > -1) {
        //   spix_idx_w = spix_idx_h + r_label_w;
        // };
        // if (spix_idx_w < num_spixels && spix_idx_w > -1) {
        //   top_data[index] = float(spix_idx_w);
        // } else {
        //   top_data[index] = float(spix_idx_h);
        // }
      }
  }


/*
Forward GPU function
*/
template <typename Dtype>
void RelToAbsIndexLayer<Dtype>::Forward_gpu(
const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  caffe_gpu_set(top[0]->count(), (Dtype)0., top[0]->mutable_gpu_data());

  const Dtype* rel_label_data = bottom[0]->gpu_data();
  const Dtype* spixel_init_data = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  const int nthreads = num_ * height_ * width_;

  RelToAbsIndexForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
    CAFFE_CUDA_NUM_THREADS>>>(nthreads, rel_label_data, spixel_init_data,
                              num_spixels_, num_spixels_w_,
                              top_data);
}

/*
Backward GPU function (NOT_IMPLEMENTED for now)
 */
template <typename Dtype>
void RelToAbsIndexLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_LAYER_GPU_FUNCS(RelToAbsIndexLayer);

}  // namespace caffe
