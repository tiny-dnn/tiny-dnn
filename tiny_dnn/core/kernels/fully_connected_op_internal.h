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
                                        const Tensor<float_t, S2> &weights,
                                        const Tensor<float_t, S3> &bias,
                                        Tensor<float_t, S4> &out_data,
                                        const fully_params &params,
                                        const bool layer_parallelize) {
  for_i(layer_parallelize, in_data.shape()[0], [&](int sample) {
    for (serial_size_t i = 0; i < params.out_size_; i++) {
      out_data.host_at(sample, i) = float_t{0};
      for (serial_size_t c = 0; c < params.in_size_; c++) {
        out_data.host_at(sample, i) +=
          weights.host_at(0, c * params.out_size_ + i) *
          in_data.host_at(sample, c);
      }

      if (params.has_bias_) {
        out_data.host_at(sample, i) += bias.host_at(0, i);
      }
    }
  });
}

template <typename S1,
          typename S2,
          typename S3,
          typename S4,
          typename S5,
          typename S6>
inline void fully_connected_op_internal(const Tensor<float_t, S1> &prev_out,
                                        const Tensor<float_t, S2> &weigths,
                                        Tensor<float_t, S3> &weights_grads,
                                        Tensor<float_t, S4> &bias_grads,
                                        Tensor<float_t, S5> &curr_delta,
                                        Tensor<float_t, S6> &prev_delta,
                                        const fully_params &params,
                                        const bool layer_parallelize) {
  for (serial_size_t sample = 0; sample < prev_out.shape()[0]; sample++) {
    for (serial_size_t c = 0; c < params.in_size_; c++) {
      // propagate delta to previous layer
      // prev_delta[c] += current_delta[r] * W_[c * out_size_ + r]
      prev_delta.host_at(sample, c) += vectorize::dot(
        curr_delta.host_iter(sample, 0),
        weigths.host_iter(0, c * params.out_size_), params.out_size_);
    }

    for_(layer_parallelize, 0, size_t(params.out_size_),
         [&](const blocked_range &r) {
           // accumulate weight-step using delta
           // dW[c * out_size + i] += current_delta[i] * prev_out[c]
           for (serial_size_t c = 0; c < params.in_size_; c++) {
             vectorize::muladd(curr_delta.host_iter(sample, r.begin()),
                               prev_out.host_at(sample, c), r.end() - r.begin(),
                               weights_grads.host_iter(
                                 sample, c * params.out_size_ + r.begin()));
           }

           if (params.has_bias_) {
             // vec_t& db = *in_grad[2];
             for (size_t i = r.begin(); i < r.end(); i++) {
               bias_grads.host_at(sample, i) += curr_delta.host_at(sample, i);
             }
           }
         });
  }
}

}  // namespace kernels
}  // namespace tiny_dnn
