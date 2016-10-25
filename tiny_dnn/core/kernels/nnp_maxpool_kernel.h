// Copyright (c) 2016, Taiga Nomi, Edgar Riba. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#pragma once

#ifdef CNN_USE_NNPACK
#include "nnpack.h"
#endif

namespace tiny_dnn {
namespace core {
namespace kernels {

inline void nnp_maxpool_kernel(const maxpool_params& params, const tensor_t& in,
                               tensor_t& a) {
#ifdef CNN_USE_NNPACK

  const cnn_size_t input_channels = params.in_.depth_;
  const cnn_size_t output_channels = params.out_.depth_;

  const nnp_size input_size = {static_cast<size_t>(params.in_.width_),
                               static_cast<size_t>(params.in_.height_)};

  const nnp_padding input_padding = {
      static_cast<size_t>(0),  // top
      static_cast<size_t>(0),  // right
      static_cast<size_t>(0),  // bottom
      static_cast<size_t>(0)   // left
  };

  const nnp_size pooling_size = {static_cast<size_t>(params.pool_size_),
                                 static_cast<size_t>(params.pool_size_)};

  const nnp_size pooling_stride = {static_cast<size_t>(params.stride_),
                                   static_cast<size_t>(params.stride_)};

  const float* input_ptr = reinterpret_cast<const float*>(&in[0]);
  float* output_ptr = reinterpret_cast<float*>(&a[0]);

  // TODO: embed it into a class
  const size_t num_mkl_threads = 1;
  pthreadpool_t threadpool = pthreadpool_create(num_mkl_threads);

  const auto status = nnp_max_pooling_output(
      input_channels, output_channels, input_size, input_padding, pooling_size,
      pooling_stride, input_ptr, output_ptr, threadpool);

  if (status != nnp_status_success) {
    throw nn_error("Could not succeed with nnp_max_pooling_output");
  }

  // TODO: embed it into a class
  pthreadpool_destroy(threadpool);
#endif
}

}  // namespace kernels
}  // namespace core
}  // namespace tiny_dnn
