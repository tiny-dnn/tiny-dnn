/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

namespace tiny_dnn {
namespace kernels {

template <typename S1, typename S2>
inline void maxpool_op_internal(
  const Tensor<float_t, S1> &in_data,
  Tensor<float_t, S2> &out_data,
  std::vector<std::vector<size_t>> &max_idx,
  const std::vector<std::vector<size_t>> &out2in,
  const bool layer_parallelize) {
  for_i(layer_parallelize, in_data.shape()[0], [&](size_t sample) {

    std::vector<size_t> &max = max_idx[sample];

    for (size_t i = 0; i < out2in.size(); i++) {
      const auto &in_index = out2in[i];
      float_t max_value    = std::numeric_limits<float_t>::lowest();
      size_t idx           = 0;
      for (auto j : in_index) {
        if (in_data.host_at(sample, j) > max_value) {
          max_value = in_data.host_at(sample, j);
          idx       = j;
        }
      }
      max[i] = idx;
      out_data.host_at(sample, i) = max_value;
    }
  });
}

template <typename S1, typename S2>
inline void maxpool_grad_op_internal(
  Tensor<float_t, S1> &prev_delta,
  const Tensor<float_t, S2> &curr_delta,
  std::vector<std::vector<size_t>> &max_idx,
  const std::vector<size_t> &in2out,
  const bool layer_parallelize) {
  for_i(layer_parallelize, prev_delta.shape()[0], [&](size_t sample) {

    const std::vector<serial_size_t> &max = max_idx[sample];

    for (size_t i = 0; i < in2out.size(); i++) {
      size_t outi = in2out[i];
      prev_delta.host_at(sample, i) = (max[outi] == i) ? curr_delta.host_at(sample, outi) : float_t{0};
    }
  });
}

}  // namespace kernels
}  // namespace tiny_dnn
