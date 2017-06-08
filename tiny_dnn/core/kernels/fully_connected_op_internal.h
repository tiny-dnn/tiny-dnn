/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/params/fully_params.h"

namespace tiny_dnn {
namespace kernels {

template <typename S1, typename S2, typename S3, typename S4>
inline void fully_connected_op_internal(const Tensor<float_t, S1> &in_data,
                                        const Tensor<float_t, S2> &W,
                                        const Tensor<float_t, S3> &bias,
                                        Tensor<float_t, S4> &out_data,
                                        const fully_params &params,
                                        const bool layer_parallelize) {
  for_i(layer_parallelize, in_data.size(), [&](int sample) {
    for (serial_size_t i = 0; i < params.out_size_; i++) {
      out_data.host_at(sample, i) = float_t{0};
      for (serial_size_t c = 0; c < params.in_size_; c++) {
        out_data.host_at(sample, i) +=
          W.host_at(c * params.out_size_ + i) * in_data.host_at(sample, i);
      }

      if (params.has_bias_) {
        out_data.host_at(sample, i) += bias.host_at(i);
      }
    }
  });
}

inline void fully_connected_op_internal(const tensor_t &prev_out,
                                        const vec_t &W,
                                        tensor_t &dW,
                                        tensor_t &db,
                                        tensor_t &curr_delta,
                                        tensor_t &prev_delta,
                                        const fully_params &params,
                                        const bool layer_parallelize) {
  for (serial_size_t sample = 0; sample < prev_out.size(); sample++) {
    for (serial_size_t c = 0; c < params.in_size_; c++) {
      // propagate delta to previous layer
      // prev_delta[c] += current_delta[r] * W_[c * out_size_ + r]
      prev_delta[sample][c] += vectorize::dot(
        &curr_delta[sample][0], &W[c * params.out_size_], params.out_size_);
    }

    for_(layer_parallelize, 0, size_t(params.out_size_),
         [&](const blocked_range &r) {
           // accumulate weight-step using delta
           // dW[c * out_size + i] += current_delta[i] * prev_out[c]
           for (serial_size_t c = 0; c < params.in_size_; c++) {
             vectorize::muladd(&curr_delta[sample][r.begin()],
                               prev_out[sample][c], r.end() - r.begin(),
                               &dW[sample][c * params.out_size_ + r.begin()]);
           }

           if (params.has_bias_) {
             // vec_t& db = *in_grad[2];
             for (size_t i = r.begin(); i < r.end(); i++) {
               db[sample][i] += curr_delta[sample][i];
             }
           }
         });
  }
}

}  // namespace kernels
}  // namespace tiny_dnn
