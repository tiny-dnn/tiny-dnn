// Copyright (c) 2016, Taiga Nomi, Edgar Riba. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#pragma once

#include "tiny_dnn/core/params/conv_params.h"

namespace tiny_dnn {
namespace core {
namespace kernels {

inline void tiny_conv2d_kernel(const conv_params& params,
                               const std::vector<const vec_t*>& in_data,
                               const vec_t& W, const vec_t& bias,
                               tensor_t& out_data,
                               const bool layer_parallelize) {
  for_i(layer_parallelize, in_data.size(), [&](int sample) {
    const vec_t& in = *in_data[sample];
    vec_t& a = out_data[sample];

    for (cnn_size_t o = 0; o < params.out.depth_; o++) {
      for (cnn_size_t inc = 0; inc < params.in.depth_; inc++) {
        if (!params.tbl.is_connected(o, inc)) continue;

        cnn_size_t idx = 0;
        idx = params.in.depth_ * o + inc;
        idx = params.weight.get_index(0, 0, idx);
        const float_t* pw = &W[idx];

        idx = params.in_padded.get_index(0, 0, inc);
        const float_t* pi = &in[idx];

        idx = params.out.get_index(0, 0, o);
        float_t* pa = &a[idx];

        for (cnn_size_t y = 0; y < params.out.height_; y++) {
          for (cnn_size_t x = 0; x < params.out.width_; x++) {
            const float_t* ppw = pw;
            const float_t* ppi =
                pi + params.in_padded.width_ * (y * params.h_stride) +
                x * params.w_stride;
            float_t sum = float_t(0);

            // should be optimized for small kernel(3x3,5x5)
            for (cnn_size_t wy = 0; wy < params.weight.height_;
                 wy++) {  // NOLINT
              for (cnn_size_t wx = 0; wx < params.weight.width_;
                   wx++) {  // NOLINT
                idx = wy * params.in_padded.width_ + wx;
                sum += *ppw++ * ppi[idx];
              }
            }
            pa[y * params.out.width_ + x] += sum;
          }
        }
      }

      if (params.has_bias) {
        float_t* pa = &a[params.out.get_index(0, 0, o)];
        float_t* paa = pa + params.out.width_ * params.out.height_;
        std::for_each(pa, paa, [&](float_t& f) { f += bias[o]; });
      }
    }
  });
}

}  // namespace kernels
}  // namespace core
}  // namespace tiny_dnn
