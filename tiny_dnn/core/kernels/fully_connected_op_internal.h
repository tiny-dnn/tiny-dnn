/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/xtensor/xarray.hpp"
#include "tiny_dnn/xtensor/xview.hpp"

#include "tiny_dnn/core/params/fully_params.h"

namespace tiny_dnn {
namespace kernels {

template <typename T>  // TODO(Randl) ??
inline void fully_connected_op_internal(const xt::xarray<float_t> &in_data,
                                        const T W,
                                        const xt::xarray<float_t> &bias,
                                        xt::xarray<float_t> &out_data,
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

template <typename T>  // TODO(Randl) ??
inline void fully_connected_op_internal(const xt::xarray<float_t> &prev_out,
                                        const T W,
                                        xt::xarray<float_t> &dW,
                                        xt::xarray<float_t> &db,
                                        xt::xarray<float_t> &curr_delta,
                                        xt::xarray<float_t> &prev_delta,
                                        const fully_params &params,
                                        const bool layer_parallelize) {
  for (serial_size_t sample = 0; sample < prev_out.size(); sample++) {
    for (serial_size_t c = 0; c < params.in_size_; c++) {
      // propagate delta to previous layer
      // prev_delta[c] += current_delta[r] * W_[c * out_size_ + r]
      prev_delta(sample, c) += vectorize::dot(
        &curr_delta(sample, 0), &W[c * params.out_size_], params.out_size_);
    }

    for (int i = 0; i < size_t(params.out_size_); ++i) {
           // accumulate weight-step using delta
           // dW[c * out_size + i] += current_delta[i] * prev_out[c]
           for (serial_size_t c = 0; c < params.in_size_; c++) {
             // TODO(Randl): return vectorization
             dW[c * params.out_size_ + i] += curr_delta[i] * prev_out[c];
             // vectorize::muladd(&curr_delta(sample, r.begin()),
             //                  prev_out(sample, c), r.end() - r.begin(),
             //                  &dW(sample, c * params.out_size_ + r.begin()));
           }

               db(sample, i) += curr_delta(sample, i);
         }
  }
}

}  // namespace kernels
}  // namespace tiny_dnn
