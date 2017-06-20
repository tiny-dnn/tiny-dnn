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

template <typename S1,
          typename S2,
          typename S3,
          typename S4,
          typename S5,
          typename S6>
inline void tiny_deconv2d_back_kernel(const Tensor<float_t, S1> &prev_out,
                                      const Tensor<float_t, S2> &weights,
                                      Tensor<float_t, S3> &weights_grads,
                                      Tensor<float_t, S4> &bias_grads,
                                      Tensor<float_t, S5> &curr_delta,
                                      Tensor<float_t, S6> &prev_delta,
                                      const deconv_params &params) {
  const float_t *W = weights.host_pbegin();
  // propagate delta to previous layer
  for_i(prev_out.size(), [&](size_t sample) {
    float_t *curr_d_begin       = curr_delta.host_pointer(sample, 0);
    float_t *prev_d_begin       = prev_delta.host_pointer(sample, 0);
    float_t *dW                 = weights_grads.host_pointer(sample, 0);
    float_t *db                 = bias_grads.host_pointer(sample, 0);
    const float_t *prev_o_begin = prev_out.host_pointer(sample, 0);
    for (serial_size_t inc = 0; inc < params.in.depth_; inc++) {
      for (serial_size_t outc = 0; outc < params.out.depth_; outc++) {
        if (!params.tbl.is_connected(outc, inc)) continue;

        serial_size_t idx = 0;
        idx               = params.in.depth_ * outc + inc;
        idx               = params.weight.get_index(0, 0, idx);
        const float_t *pw = &W[idx];

        idx                       = params.out_unpadded.get_index(0, 0, outc);
        const float_t *pdelta_src = &curr_d_begin[idx];

        idx                 = params.in.get_index(0, 0, inc);
        float_t *pdelta_dst = &prev_d_begin[idx];

        for (serial_size_t y = 0; y < params.in.height_; y++) {
          for (serial_size_t x = 0; x < params.in.width_; x++) {
            const float_t *ppw = pw;

            float_t *ppdelta_dst = pdelta_dst + y * params.in.width_ + x;
            float_t sum{0};

            for (serial_size_t wy = 0; wy < params.weight.height_; wy++) {
              for (serial_size_t wx = 0; wx < params.weight.width_; wx++) {
                idx = (y * params.h_stride + wy) * params.out.width_ +
                      (x * params.w_stride + wx);
                sum += ppw[wy * params.weight.width_ + wx] * pdelta_src[idx];
              }
            }
            *ppdelta_dst += sum;
          }
        }
      }
    }

    // accumulate dw
    for (serial_size_t inc = 0; inc < params.in.depth_; inc++) {
      for (serial_size_t outc = 0; outc < params.out.depth_; outc++) {
        if (!params.tbl.is_connected(outc, inc)) continue;

        for (serial_size_t wy = 0; wy < params.weight.height_; wy++) {
          for (serial_size_t wx = 0; wx < params.weight.width_; wx++) {
            float_t dst{0};

            serial_size_t idx    = 0;
            idx                  = params.in.get_index(0, 0, inc);
            const float_t *prevo = &prev_o_begin[idx];

            idx                  = params.out.get_index(wx, wy, outc);
            const float_t *delta = &curr_d_begin[idx];

            for (serial_size_t y = 0; y < params.in.height_; y++) {
              dst +=
                vectorize::dot(prevo + y * params.in.width_,
                               delta + y * params.out.width_, params.in.width_);
            }

            idx = params.in.depth_ * outc + inc;
            dW[params.weight.get_index(wx, wy, idx)] += dst;
          }
        }
      }
    }

    // accumulate db
    if (params.has_bias) {
      // vec_t& db = *in_grad[2];

      for (serial_size_t outc = 0; outc < params.out.depth_; outc++) {
        serial_size_t idx    = params.out.get_index(0, 0, outc);
        const float_t *delta = &curr_d_begin[idx];
        // TODO: name
        const float_t *deltaa = delta + params.out.width_ * params.out.height_;
        db[outc] += std::accumulate(delta, deltaa, float_t{0});
      }
    }
  });
}

}  // namespace kernels
}  // namespace core
}  // namespace tiny_dnn