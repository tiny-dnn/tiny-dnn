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

/**
 * Forward propogation for fully connected layer with internal backend
 * @param in_data
 * @param weights
 * @param bias
 * @param out_data
 * @param params
 * @param layer_parallelize
 */
template <typename S1, typename S2, typename S3, typename S4>
inline void fully_connected_op_internal(const Tensor<float_t, S1> &in_data,
                                        const Tensor<float_t, S2> &weights,
                                        const Tensor<float_t, S3> &bias,
                                        Tensor<float_t, S4> &out_data,
                                        const bool layer_parallelize) {
  size_t out_size = out_data.shape()[1], in_size = in_data.shape()[1];
  for_i(layer_parallelize, in_data.shape()[0], [&](int sample) {
    for (size_t i = 0; i < out_size; i++) {
      out_data.host_at(sample, i) = float_t{0};
      for (size_t c = 0; c < in_size; c++) {
        out_data.host_at(sample, i) +=
          weights.host_at(0, c * out_size + i) * in_data.host_at(sample, c);
      }

      if (bias.size() >= out_size) {
        out_data.host_at(sample, i) += bias.host_at(0, i);
      }
    }
  });
}

/**
 * Backward propogation for fully connected layer with internal backend
 * @param prev_out
 * @param weigths
 * @param weights_grads
 * @param bias_grads
 * @param curr_delta
 * @param prev_delta
 * @param params
 * @param layer_parallelize
 */
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
                                        const bool layer_parallelize) {
  size_t out_size = curr_delta.shape()[1], in_size = prev_delta.shape()[1],
         sample_num = prev_out.shape()[0];
  for (size_t sample = 0; sample < sample_num; sample++) {
    for (size_t c = 0; c < in_size; c++) {
      // propagate delta to previous layer
      // prev_delta[c] += current_delta[r] * W_[c * out_size_ + r]
      prev_delta.host_at(sample, c) +=
        vectorize::dot(curr_delta.host_pointer(sample, 0),
                       weigths.host_pointer(0, c * out_size), out_size);
    }

    for_(layer_parallelize, 0, size_t(out_size), [&](const blocked_range &r) {
      // accumulate weight-step using delta
      // dW[c * out_size + i] += current_delta[i] * prev_out[c]
      for (size_t c = 0; c < in_size; c++) {
        vectorize::muladd(
          curr_delta.host_pointer(sample, r.begin()),
          prev_out.host_at(sample, c), r.end() - r.begin(),
          weights_grads.host_pointer(sample, c * out_size + r.begin()));
      }

      if (bias_grads.size() >= out_size) {
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
