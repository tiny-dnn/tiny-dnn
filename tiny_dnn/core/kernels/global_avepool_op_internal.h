/*
    Copyright (c) 2017, Taiga Nomi
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

namespace tiny_dnn {
namespace kernels {

inline void global_avepool_op_internal(
  const tensor_t &in_data,
  tensor_t &out_data,
  const core::global_avepool_params &params,
  const bool layer_parallelize) {
  for_i(layer_parallelize, in_data.size(), [&](size_t sample) {
    const vec_t &in = in_data[sample];
    vec_t &out      = out_data[sample];

    const size_t pool_area = params.in.width_ * params.in.height_;
    for (size_t i = 0; i < params.in.depth_; i++) {
      for (size_t j = 0; j < pool_area; j++) {
        out[i] += in[i * pool_area + j];
      }
      out[i] /= pool_area;
    }
  });
}

inline void global_avepool_grad_op_internal(
  tensor_t &prev_delta,
  const tensor_t &curr_delta,
  const core::global_avepool_params &params,
  const bool layer_parallelize) {
  for_i(layer_parallelize, prev_delta.size(), [&](size_t sample) {
    vec_t &prev       = prev_delta[sample];
    const vec_t &curr = curr_delta[sample];

    const size_t pool_area = params.in.width_ * params.in.height_;
    for (size_t i = 0; i < params.in.depth_; i++) {
      const float_t pi = curr[i] / pool_area;
      for (size_t j = 0; j < pool_area; j++) {
        prev[i * pool_area + j] = pi;
      }
    }
  });
}

}  // namespace kernels
}  // namespace tiny_dnn
