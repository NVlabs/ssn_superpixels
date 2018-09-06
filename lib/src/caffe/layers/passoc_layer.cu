/*
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Author: Varun Jampani
*/

#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/layers/passoc_layer.hpp"

#include <cmath>


namespace caffe {

template <typename Dtype>
__global__ void PassocForwardGPU(const int nthreads, const Dtype* bottom_data,
  const Dtype* spixel_data, const Dtype* index_data,
  const int in_dim, const int height,
  const int width, const int num_spixels, const int num_spixels_h,
  const int num_spixels_w, const float scale_value, Dtype* top_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {

      const int spatial_dim = height * width;
      const int n = index / (spatial_dim * 9);
      const int s = index % (spatial_dim * 9);

      const int c_index = s / spatial_dim;
      const int p_index = s % spatial_dim;

      const int init_spix_index = static_cast<int>(index_data[n * spatial_dim + p_index]);
      int spixel_idx = init_spix_index;

      // Convert spixel_idx based on the association channel
      const int r_idx = c_index;
      const int r_idx_h = r_idx / 3 - 1;
      int r_idx_w = r_idx % 3 - 1;

      bool invalid_spixel = false;

      int spix_idx_h = init_spix_index + r_idx_h * num_spixels_w;
      if (spix_idx_h >= num_spixels || spix_idx_h <= -1) {
        spix_idx_h = init_spix_index;
        invalid_spixel = true;
      }

      if (((spix_idx_h + 1) % num_spixels_w) == 0 && r_idx_w == 1) {
        r_idx_w = 0;
        invalid_spixel = true;
      } else if ((spix_idx_h % num_spixels_w) == 0 && r_idx_w == -1) {
        r_idx_w = 0;
        invalid_spixel = true;
      }

      int spix_idx_w = spix_idx_h + r_idx_w;
      if (spix_idx_w < num_spixels && spix_idx_w > -1) {
        spixel_idx = spix_idx_w;
      } else {
        spixel_idx = spix_idx_h;
        invalid_spixel = true;
      }

      Dtype sq_dist = 0;
      if (invalid_spixel == true) {
        sq_dist = 10000.0;
      } else {
        for (int k = 0; k < in_dim; k++) {
          int spixel_offset = ((n * in_dim + k) * num_spixels + spixel_idx);
          int bottom_offset = ((n * in_dim + k) * spatial_dim + p_index);
          sq_dist += pow(bottom_data[bottom_offset] - spixel_data[spixel_offset], 2);
        }
      }

      int top_offset = ((n * 9 + c_index) * spatial_dim + p_index);
      top_data[top_offset] = sq_dist * scale_value;
    }
}

// ((n * channels() + c) * height() + h) * width() + w;

template <typename Dtype>
void PassocLayer<Dtype>::Forward_gpu(
const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  caffe_gpu_set(top[0]->count(), (Dtype)0., top[0]->mutable_gpu_data());

  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* spixel_data = bottom[1]->gpu_data();
  const Dtype* index_data = bottom[2]->gpu_data();

  Dtype* top_data = top[0]->mutable_gpu_data();

  const int nthreads = num_ * height_ * width_ * 9;
  // NOLINT_NEXT_LINE(whitespace/operators)
  PassocForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
    CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_data, spixel_data, index_data,
                              channels_, height_, width_,
                              num_spixels_, num_spixels_h_, num_spixels_w_,
                              scale_value_, top_data);
}



template <typename Dtype>
__global__ void PassocBackwardBottomGPU(const int nthreads,
  const Dtype* top_diff, const Dtype* bottom_data,
  const Dtype* spixel_data, const Dtype* index_data,
  const int in_dim, const int height, const int width,
  const int num_spixels, const int num_spixels_h,
  const int num_spixels_w, const float scale_value,
  Dtype* bottom_diff, Dtype* spixel_diff) {
    CUDA_KERNEL_LOOP(index, nthreads) {

      const int spatial_dim = height * width;
      const int n = index / (spatial_dim * 9);
      const int s = index % (spatial_dim * 9);

      const int c_index = s / spatial_dim;
      const int p_index = s % spatial_dim;

      const int init_spix_index = static_cast<int>(index_data[n * spatial_dim + p_index]);
      int spixel_idx = init_spix_index;

      // Convert spixel_idx based on the association channel
      const int r_idx = c_index;
      const int r_idx_h = r_idx / 3 - 1;
      int r_idx_w = r_idx % 3 - 1;

      bool invalid_spixel = false;

      int spix_idx_h = init_spix_index + r_idx_h * num_spixels_w;
      if (spix_idx_h >= num_spixels || spix_idx_h <= -1) {
        spix_idx_h = init_spix_index;
        invalid_spixel = true;
      }

      if (((spix_idx_h + 1) % num_spixels_w) == 0 && r_idx_w == 1) {
        r_idx_w = 0;
        invalid_spixel = true;
      } else if ((spix_idx_h % num_spixels_w) == 0 && r_idx_w == -1) {
        r_idx_w = 0;
        invalid_spixel = true;
      }

      int spix_idx_w = spix_idx_h + r_idx_w;
      if (spix_idx_w < num_spixels && spix_idx_w > -1) {
        spixel_idx = spix_idx_w;
      } else {
        spixel_idx = spix_idx_h;
        invalid_spixel = true;
      }

      int top_offset = ((n * 9 + c_index) * spatial_dim + p_index);

      for (int k = 0; k < in_dim; k++) {
        int spixel_offset = ((n * in_dim + k) * num_spixels + spixel_idx);
        int bottom_offset = ((n * in_dim + k) * spatial_dim + p_index);

        if (invalid_spixel) {
          caffe_gpu_atomic_add((Dtype) 0.0,
              bottom_diff + bottom_offset);

          caffe_gpu_atomic_add((Dtype) 0.0,
              spixel_diff + spixel_offset);
        } else {
          caffe_gpu_atomic_add((Dtype) top_diff[top_offset] *
            2 * scale_value * (bottom_data[bottom_offset] - spixel_data[spixel_offset]),
              bottom_diff + bottom_offset);

          caffe_gpu_atomic_add((Dtype) top_diff[top_offset] *
            2 * scale_value * (spixel_data[spixel_offset] - bottom_data[bottom_offset]),
              spixel_diff + spixel_offset);
        }
      }
    }
}

/*
Backward GPU function
 */
template <typename Dtype>
void PassocLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[2]) {
      LOG(FATAL) << this->type()
                 << " Layer cannot backpropagate to spixel index inputs.";
    }
    if (propagate_down[0] || propagate_down[1]) {
      caffe_gpu_set(bottom[0]->count(), (Dtype)0.,
        bottom[0]->mutable_gpu_diff());
      caffe_gpu_set(bottom[1]->count(), (Dtype)0.,
        bottom[1]->mutable_gpu_diff());

      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      Dtype* spixel_diff = bottom[1]->mutable_gpu_diff();

      const Dtype* top_diff = top[0]->gpu_diff();
      const Dtype* bottom_data = bottom[0]->gpu_data();
      const Dtype* spixel_data = bottom[1]->gpu_data();
      const Dtype* index_data = bottom[2]->gpu_data();

      const int nthreads = num_ * height_ * width_ * 9;
      // NOLINT_NEXT_LINE(whitespace/operators)
      PassocBackwardBottomGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_diff,
                                  bottom_data, spixel_data, index_data,
                                  channels_, height_, width_,
                                  num_spixels_, num_spixels_h_,
                                  num_spixels_w_, scale_value_,
                                  bottom_diff, spixel_diff);
    }
}


INSTANTIATE_LAYER_GPU_FUNCS(PassocLayer);

}  // namespace caffe
