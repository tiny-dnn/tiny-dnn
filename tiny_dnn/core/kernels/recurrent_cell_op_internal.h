/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/params/recurrent_cell_params.h"

namespace tiny_dnn {
namespace kernels {

inline void recurrent_cell_op_internal(const tensor_t &in_data,
                                       const tensor_t &prev_h,
                                       const vec_t &U,
                                       const vec_t &W,
                                       const vec_t &V,
                                       const vec_t &bias,
                                       const vec_t &c,
                                       tensor_t &out_data,
                                       tensor_t &out_h,
                                       const recurrent_cell_params &params,
                                       const bool layer_parallelize) {
  for_i(layer_parallelize, in_data.size(), [&](size_t sample) {
    const vec_t &in         = in_data[sample];
    const vec_t &prev_state = prev_h[sample];
    vec_t &out              = out_data[sample];
    vec_t &next_state       = out_h[sample];

    for (size_t o = 0; o < params.out_size_; o++) {
      out[o] = float_t{0};

      // W * h(t-1)
      for (size_t o_2 = 0; o_2 < params.out_size_; o_2++) {
        next_state[o] += W[o_2 * params.out_size_ + o] * prev_state[o_2];
      }

      // U*x(t)
      for (size_t i = 0; i < params.in_size_; i++) {
        next_state[o] += U[i * params.out_size_ + o] * in[i];
      }

      if (params.has_bias_) {
        next_state[o] += bias[o];
      }
    }

    params.activation_->forward_activation(next_state, next_state);

    // V matrix is out_size_ x out_size_
    for (size_t o = 0; o < params.out_size_; o++) {
      for (size_t o_2 = 0; o_2 < params.out_size_; o_2++) {
        out[o] += V[o_2 * params.out_size_ + o] * next_state[o_2];
      }

      if (params.has_bias_) {
        out[o] += c[o];
      }
    }

  });
}

inline void recurrent_cell_op_internal(const tensor_t &prev_out,
                                       const tensor_t &prev_h,
                                       const vec_t &U,
                                       const vec_t &W,
                                       const vec_t &V,
                                       tensor_t &dU,
                                       tensor_t &dW,
                                       tensor_t &dV,
                                       tensor_t &db,
                                       tensor_t &dc,
                                       tensor_t &curr_output_delta,
                                       tensor_t &curr_state_delta,
                                       tensor_t &prev_output_delta,
                                       tensor_t &prev_state_delta,
                                       const tensor_t &out_h,
                                       const recurrent_cell_params &params,
                                       const bool layer_parallelize) {
  for (size_t sample = 0; sample < prev_out.size(); sample++) {
    // from output to h
    for (size_t o = 0; o < params.out_size_; o++) {
      // propagate delta from output to h.
      curr_state_delta[sample][o] +=
        vectorize::dot(&curr_output_delta[sample][0], &V[o * params.out_size_],
                       params.out_size_);
    }

    // h'(t)
    params.activation_->backward_activation(prev_h[sample], out_h[sample],
                                            curr_state_delta[sample],
                                            curr_state_delta[sample]);

    // \delta h(t) -W-> h(t-1)
    for (size_t o = 0; o < params.out_size_; o++) {
      prev_state_delta[sample][o] +=
        vectorize::dot(&curr_state_delta[sample][0], &W[o * params.out_size_],
                       params.out_size_);
    }

    // \delta h(t) -U-> \delta x(t)
    for (size_t i = 0; i < params.in_size_; i++) {
      prev_output_delta[sample][i] +=
        vectorize::dot(&curr_state_delta[sample][0], &U[i * params.out_size_],
                       params.out_size_);
    }

    for_(
      layer_parallelize, 0, size_t(params.out_size_),
      [&](const blocked_range &r) {
        // accumulate weight-step using delta
        // dW[c * out_size + i] += current_delta[i] * prev_out[o]
        const size_t begin  = r.begin();
        const size_t end    = r.end();
        const size_t stride = end - begin;
        for (size_t o = 0; o < params.out_size_; o++) {
          vectorize::muladd(&curr_output_delta[sample][begin], out_h[sample][o],
                            stride, &dV[sample][o * params.out_size_ + begin]);
        }

        if (params.has_bias_) {
          // vec_t& dc;
          for (size_t o = begin; o < end; o++) {
            dc[sample][o] += curr_output_delta[sample][o];
          }
        }

        for (size_t o = 0; o < params.out_size_; o++) {
          vectorize::muladd(&curr_state_delta[sample][begin], prev_h[sample][o],
                            stride, &dW[sample][o * params.out_size_ + begin]);
        }

        for (size_t i = 0; i < params.in_size_; i++) {
          vectorize::muladd(&curr_state_delta[sample][begin],
                            prev_out[sample][i], stride,
                            &dU[sample][i * params.out_size_ + begin]);
        }

        if (params.has_bias_) {
          // vec_t& db;
          for (size_t o = begin; o < end; o++) {
            db[sample][o] += curr_state_delta[sample][o];
          }
        }

      });
  }
}

}  // namespace kernels
}  // namespace tiny_dnn
