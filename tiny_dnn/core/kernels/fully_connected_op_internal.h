/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/xtensor/xarray.hpp"
#include "tiny_dnn/xtensor/xexpression.hpp"
#include "tiny_dnn/xtensor/xview.hpp"

#include "tiny_dnn/core/params/fully_params.h"

namespace tiny_dnn {
namespace kernels {

template <class E1, class E2, class E3, class E4>
inline void fully_connected_op_internal(const E1 &in_data,
                                        const E2 &W,
                                        const E3 &bias,
                                        E4 &out_data,
                                        const fully_params &params,
                                        const bool layer_parallelize) {
  for_i(layer_parallelize, in_data.shape()[0], [&](int sample) {

    auto const in = xt::view(in_data, sample, xt::all());
    auto out      = xt::view(out_data, sample, xt::all());

    for (serial_size_t i = 0; i < params.out_size_; i++) {
      out[i] = float_t{0};
      for (serial_size_t c = 0; c < params.in_size_; c++) {
        out[i] += W[c * params.out_size_ + i] * in[c];
      }

      if (params.has_bias_) {
        out[i] += bias[i];
      }
    }
  });
}

// TODO(Randl): enable on xexpressions and views?
// TODO(Randl): rvalue?
template <class E1, class E2, class E3, class E4, class E5, class E6>
inline void fully_connected_op_internal(const E1 &prev_out,
                                        const E2 &W,
                                        E3 &dW,
                                        E4 &db,
                                        E5 &curr_delta,
                                        E6 &prev_delta,
                                        const fully_params &params,
                                        const bool layer_parallelize) {
  for (serial_size_t sample = 0; sample < prev_out.shape()[0]; sample++) {
    for (serial_size_t c = 0; c < params.in_size_; c++) {
      // propagate delta to previous layer
      // prev_delta[c] += current_delta[r] * W_[c * out_size_ + r]
      prev_delta(sample, c) += vectorize::dot(
        &curr_delta(sample, 0), &W(c * params.out_size_), params.out_size_);
    }

    for_(layer_parallelize, 0, size_t(params.out_size_),
         [&](const blocked_range &r) {
           // accumulate weight-step using delta
           // dW[c * out_size + i] += current_delta[i] * prev_out[c]
           for (serial_size_t c = 0; c < params.in_size_; c++) {
             vectorize::muladd(&curr_delta(sample, r.begin()),
                               prev_out(sample, c), r.end() - r.begin(),
                               &dW(sample, c * params.out_size_ + r.begin()));
           }

           if (params.has_bias_) {
             // vec_t& db = *in_grad[2];
             for (size_t i = r.begin(); i < r.end(); i++) {
               db(sample, i) += curr_delta(sample, i);
             }
           }
         });
  }
  // db = db+curr_delta;
  // FIXME: No tests found o_O
}

}  // namespace kernels
}  // namespace tiny_dnn
