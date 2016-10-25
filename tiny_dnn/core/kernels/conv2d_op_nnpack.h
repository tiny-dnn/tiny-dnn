// Copyright (c) 2016, Taiga Nomi, Edgar Riba. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#pragma once

#include "tiny_dnn/core/params/conv_params.h"

#ifdef CNN_USE_NNPACK
#include "nnpack.h"

inline nnp_convolution_algorithm nnp_algorithm() {
  return nnp_convolution_algorithm_auto;
}

inline nnp_convolution_kernel_transform_strategy nnp_kts() {
  return nnp_convolution_kernel_transform_strategy_reuse;
}
#endif

namespace tiny_dnn {
namespace kernels {

inline void conv2d_op_nnpack(const tensor_t& in_data, const vec_t& W,
                             const vec_t& bias, tensor_t& out_data,
                             const core::conv_params& params) {
#ifdef CNN_USE_NNPACK
  nnp_status init_status = nnp_initialize();
  if (init_status != nnp_status_success) {
    throw nn_error("Cannot initialize NNPACK.");
  }

  // TOOD: use input config
  const auto algorithm = nnp_algorithm();
  const auto kernel_transform_strategy = nnp_kts();

  const cnn_size_t input_channels = params.in.depth_;
  const cnn_size_t output_channels = params.out.depth_;

  const nnp_size input_size = {static_cast<size_t>(params.in.width_),
                               static_cast<size_t>(params.in.height_)};

  const nnp_size kernel_size = {static_cast<size_t>(params.weight.width_),
                                static_cast<size_t>(params.weight.height_)};

  const float_t dx = params.in_padded.width_ - params.in.width_;
  const float_t dy = params.in_padded.height_ - params.in.height_;

  // we'll assume that padding is symmetric

  const nnp_padding padding = {
      static_cast<size_t>(dy / 2),  // top
      static_cast<size_t>(dx / 2),  // right
      static_cast<size_t>(dy / 2),  // bottom
      static_cast<size_t>(dx / 2)   // left
  };

  const float* input_ptr = reinterpret_cast<const float*>(&in_data[0]);
  const float* kernel_ptr = reinterpret_cast<const float*>(&W[0]);
  const float* bias_ptr = reinterpret_cast<const float*>(&bias[0]);

  float* output_ptr = reinterpret_cast<float*>(&out_data[0]);

  // TODO: embed it into a class
  const size_t num_mkl_threads = 1;
  pthreadpool_t threadpool = pthreadpool_create(num_mkl_threads);

  nnp_profile* profile = nullptr;

  nnp_status status = nnp_convolution_inference(
      algorithm, kernel_transform_strategy, input_channels, output_channels,
      input_size, padding, kernel_size, input_ptr, kernel_ptr, bias_ptr,
      output_ptr, threadpool, profile);

  if (status != nnp_status_success) {
    throw nn_error("Could not succeed with nnp_convolution_inference");
  }

  // TODO: embed it into a class
  pthreadpool_destroy(threadpool);
#endif
}

}  // namespace kernels
}  // namespace tiny_dnn
