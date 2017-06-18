/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/params/deconv_params.h"

namespace tiny_dnn {
namespace core {
namespace kernels {

// TODO: match structure with other kernels (op class, etc.)
template <typename S1, typename S2, typename S3, typename S4>
inline void tiny_deconv2d_kernel(const Tensor<float_t, S1> &in_data,
                                 const Tensor<float_t, S2> &weights,
                                 const Tensor<float_t, S3> &bias,
                                 Tensor<float_t, S4> &out_data,
                                 const deconv_params &params,
                                 const bool layer_parallelize) {
  size_t num_of_samples = in_data.shape()[0];
  const float_t *W      = weights.host_pointer(0, 0);  // TODO(Randl)
  for_i(layer_parallelize, num_of_samples, [&](size_t sample) {
    const float_t *in = in_data.host_pointer(sample, 0);
    float_t *out      = out_data.host_pointer(sample, 0);
    for (size_t o = 0; o < params.out.depth_; o++) {
      for (size_t inc = 0; inc < params.in.depth_; inc++) {
        if (!params.tbl.is_connected(o, inc)) continue;

        size_t idx = 0;
        idx               = params.in.depth_ * o + inc;
        idx               = params.weight.get_index(0, 0, idx);
        assert(idx < weights.size());
        const float_t *pw = &W[idx];

        idx = params.in.get_index(0, 0, inc);
        assert(sample < num_of_samples && idx <= in_data.shape()[1]);
        const float_t *pi = &in[idx];

        idx = params.out.get_index(0, 0, o);
        assert(sample < out_data.shape()[0] && idx <= out_data.shape()[1]);
        float_t *pout = &out[idx];

        for (size_t y = 0; y < params.in.height_; y++) {
          for (size_t x = 0; x < params.in.width_; x++) {
            const float_t *ppw = pw;
            const float_t *ppi = pi + y * params.in.width_ + x;
            // should be optimized for small kernel(3x3,5x5)
            for (size_t wy = 0; wy < params.weight.height_; wy++) {
              for (size_t wx = 0; wx < params.weight.width_; wx++) {
                pout[(y * params.h_stride + wy) * params.out.width_ +
                     (x * params.w_stride + wx)] +=
                  ppw[wy * params.weight.width_ + wx] * (*ppi);
              }
            }
          }
        }
      }

      if (params.has_bias) {
        float_t *pout =
          out_data.host_pointer(sample, params.out.get_index(0, 0, o));
        float_t *pout2 = pout + params.out.width_ * params.out.height_;
        std::for_each(pout, pout2, [&](float_t &f) { f += bias.host_at(o); });
      }
    }
  });
}

}  // namespace kernels
}  // namespace core
}  // namespace tiny_dnn
