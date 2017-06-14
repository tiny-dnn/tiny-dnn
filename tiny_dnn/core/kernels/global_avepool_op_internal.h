/*
    Copyright (c) 2017, Taiga Nomi
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

namespace tiny_dnn {
namespace kernels {

template <typename S1, typename S2>
inline void global_avepool_op_internal(
  const Tensor<float_t, S1> &in_data,
  Tensor<float_t, S2> &out_data,
  const core::global_avepool_params &params,
  const bool layer_parallelize) {
  for_i(layer_parallelize, in_data.size(), [&](size_t sample) {
    auto in  = in_data.host_pointer(sample, 0);
    auto out = out_data.host_pointer(sample, 0);

    const size_t pool_area = params.in.width_ * params.in.height_;
    for (size_t i = 0; i < params.in.depth_; i++) {
      for (size_t j = 0; j < pool_area; j++) {
        out[i] += in[i * pool_area + j];
      }
      out[i] /= pool_area;
    }
  });
}

template <typename S1, typename S2>
inline void global_avepool_grad_op_internal(
  Tensor<float_t, S1> &prev_delta,
  const Tensor<float_t, S2> &curr_delta,
  const core::global_avepool_params &params,
  const bool layer_parallelize) {
  for_i(layer_parallelize, prev_delta.size(), [&](size_t sample) {
    auto prev = prev_delta.host_pointer(sample, 0);
    auto curr = curr_delta.host_pointer(sample, 0);

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
