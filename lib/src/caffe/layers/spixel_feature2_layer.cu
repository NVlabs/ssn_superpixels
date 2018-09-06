/*
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Author: Varun Jampani
*/

#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/layers/spixel_feature2_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SpixelFeature2ForwardGPU(const int nthreads,
  const Dtype* bottom_data, const Dtype* assoc_data,
  const Dtype* index_data, const int in_dim, const int height, const int width,
  const int num_spixels, const int num_spixels_h, const int num_spixels_w,
  Dtype* top_data, Dtype* weight_data) {

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

      // int assoc_offset = ((n * in_dim + c_index) * spatial_dim + p_index);
      int assoc_offset = ((n * 9 + c_index) * spatial_dim + p_index);

      if (invalid_spixel == false){
        for (int k = 0; k < in_dim; k++) {
          int top_offset = ((n * in_dim + k) * num_spixels + spixel_idx);
          if (k < in_dim) {
            int bottom_offset = ((n * in_dim + k) * spatial_dim + p_index);
            caffe_gpu_atomic_add((Dtype) bottom_data[bottom_offset] * assoc_data[assoc_offset],
              top_data + top_offset);
          }
        }

        int weight_offset = (n * num_spixels + spixel_idx);
        caffe_gpu_atomic_add(assoc_data[assoc_offset],
          weight_data + weight_offset);
      }
    }
}

template <typename Dtype>
__global__ void SpixelFeature2AverageForwardGPU(const int nthreads,
  const int num_spixels, const int out_dim,
  Dtype* top_data, Dtype* weight_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      const int n = index / num_spixels;
      const int s = index % num_spixels;

      const int weight_offset = (n * num_spixels + s);
      for (int k = 0; k < out_dim; k++) {
        const int top_offset = ((n * out_dim + k) * num_spixels + s);
        if (weight_data[weight_offset] < 0.001) {
          top_data[top_offset] = 0;
        } else {
          top_data[top_offset] /= weight_data[weight_offset];
        }
      }
    }
}


/*
Forward GPU function
*/
template <typename Dtype>
void SpixelFeature2Layer<Dtype>::Forward_gpu(
const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  caffe_gpu_set(top[0]->count(), (Dtype)0., top[0]->mutable_gpu_data());
  caffe_gpu_set(spixel_weights_.count(), (Dtype)0.,
    spixel_weights_.mutable_gpu_data());

  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* assoc_data = bottom[1]->gpu_data();
  const Dtype* index_data = bottom[2]->gpu_data();

  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* weight_data = spixel_weights_.mutable_gpu_data();

  const int nthreads = num_ * height_ * width_ * 9;
  // NOLINT_NEXT_LINE(whitespace/operators)
  SpixelFeature2ForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
    CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_data, assoc_data, index_data,
                              in_channels_, height_, width_,
                              num_spixels_, num_spixels_h_, num_spixels_w_,
                              top_data, weight_data);

  const int nthreads2 = num_ * num_spixels_;
  // NOLINT_NEXT_LINE(whitespace/operators)
  SpixelFeature2AverageForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads2),
    CAFFE_CUDA_NUM_THREADS>>>(nthreads2, num_spixels_,
                              in_channels_,
                              top_data, weight_data);
}


template <typename Dtype>
__global__ void SpixelFeature2BackwardBottomGPU(const int nthreads,
  const Dtype* top_diff, const Dtype* top_data, const Dtype* bottom_data,
  const Dtype* assoc_data, const Dtype* index_data,
  const int in_dim, const int height, const int width,
  const int num_spixels, const int num_spixels_h, const int num_spixels_w,
  Dtype* bottom_diff, Dtype* assoc_diff, const Dtype* weight_data) {
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

      int assoc_offset = ((n * 9 + c_index) * spatial_dim + p_index);
      int weight_offset = (n * num_spixels + spixel_idx);

      for (int k = 0; k < in_dim; k++) {
        int top_offset = ((n * in_dim + k) * num_spixels + spixel_idx);
        if (k < in_dim) {
          int bottom_offset = ((n * in_dim + k) * spatial_dim + p_index);
          if (weight_data[weight_offset] > 0.001) {
            if (invalid_spixel) {
              caffe_gpu_atomic_add((Dtype) 0.0, bottom_diff + bottom_offset);
              caffe_gpu_atomic_add((Dtype) 0.0, assoc_diff + assoc_offset);
            } else {
              caffe_gpu_atomic_add((Dtype) top_diff[top_offset] * assoc_data[assoc_offset] / weight_data[weight_offset],
                bottom_diff + bottom_offset);
              caffe_gpu_atomic_add((Dtype) top_diff[top_offset] * ((bottom_data[bottom_offset] - top_data[top_offset]) / weight_data[weight_offset]),
                assoc_diff + assoc_offset);
            }
          }
        }
      }
    }
}


/*
Backward GPU function
 */
template <typename Dtype>
void SpixelFeature2Layer<Dtype>::Backward_gpu(
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
      Dtype* assoc_diff = bottom[1]->mutable_gpu_diff();

      const Dtype* top_diff = top[0]->gpu_diff();
      const Dtype* top_data = top[0]->gpu_data();
      const Dtype* bottom_data = bottom[0]->gpu_data();
      const Dtype* assoc_data = bottom[1]->gpu_data();
      const Dtype* index_data = bottom[2]->gpu_data();
      const Dtype* weight_data = spixel_weights_.mutable_gpu_data();

      const int nthreads = num_ * height_ * width_ * 9;
      // NOLINT_NEXT_LINE(whitespace/operators)
      SpixelFeature2BackwardBottomGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_diff, top_data,
                                  bottom_data, assoc_data, index_data,
                                  in_channels_, height_, width_,
                                  num_spixels_, num_spixels_h_, num_spixels_w_,
                                  bottom_diff, assoc_diff, weight_data);
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(SpixelFeature2Layer);

}  // namespace caffe
