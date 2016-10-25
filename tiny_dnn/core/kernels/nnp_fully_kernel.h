// Copyright (c) 2016, Taiga Nomi, Edgar Riba. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#pragma once

#include "tiny_dnn/core/params/fully_params.h"

namespace tiny_dnn {
namespace core {
namespace kernels {

inline void nnp_fully_connected_kernel(const fully_params& params,
                                       const tensor_t& in, const vec_t& W,
                                       vec_t& b, tensor_t& a,
                                       const bool layer_parallelize) {
#ifdef CNN_USE_NNPACK
  const float* kernel_ptr = reinterpret_cast<const float*>(&W[0]);
  const float* input_ptr = reinterpret_cast<const float*>(&in[0]);
  float* output_ptr = reinterpret_cast<float*>(&a[0]);

  // TODO: embed it into a class
  const size_t num_mkl_threads = 1;
  pthreadpool_t threadpool = pthreadpool_create(num_mkl_threads);

  const auto status = nnp_fully_connected_inference(
      params.in_size_, params.out_size_, input_ptr, kernel_ptr, output_ptr,
      threadpool);

  if (status != nnp_status_success) {
    throw nn_error("Could not succeed with nnp_max_pooling_output");
  }

  // TODO: embed it into a class
  pthreadpool_destroy(threadpool);

  // TODO: find a proper way to do this
  if (params.has_bias_) {
    for_i(layer_parallelize, params.out_size_, [&](int i) {
      // TODO(edgar): revise this
      // a[i] += b[i];
    });
  }
#endif
}

}  // namespace kernels
}  // namespace core
}  // namespace tiny_dnn
