/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/params/deconv_params.h"

namespace tiny_dnn {
namespace kernels {

template <typename S1, typename S2, typename S3, typename S4>
inline void deconv2d_op_internal(const Tensor<float_t, S1> &in_data,
                                 const Tensor<float_t, S2> &weights,
                                 const Tensor<float_t, S3> &bias,
                                 Tensor<float_t, S4> &out_data,
                                 const deconv_params &params,
                                 const bool &has_bias,
                                 const bool &parallelize) {
  size_t num_of_samples = in_data.shape()[0];
  const float_t *W      = weights.host_pbegin();
  for_i(parallelize, num_of_samples, [&](size_t sample) {
    const float_t *in = in_data.host_pointer(sample, 0);
    float_t *out      = out_data.host_pointer(sample, 0);
    for (size_t o = 0; o < params.out.depth_; o++) {
      for (size_t inc = 0; inc < params.in.depth_; inc++) {
        if (!params.tbl.is_connected(o, inc)) continue;

        size_t idx = 0;
        idx        = params.in.depth_ * o + inc;
        idx        = params.weight.get_index(0, 0, idx);
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

      if (has_bias) {
        float_t *pout =
          out_data.host_pointer(sample, params.out.get_index(0, 0, o));
        float_t *pout2 = pout + params.out.width_ * params.out.height_;
        std::for_each(pout, pout2, [&](float_t &f) { f += bias.host_at(o); });
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
inline void deconv2d_op_internal(const Tensor<float_t, S1> &prev_out,
                                 const Tensor<float_t, S2> &weights,
                                 Tensor<float_t, S3> &weights_grads,
                                 Tensor<float_t, S4> &bias_grads,
                                 Tensor<float_t, S5> &curr_delta,
                                 Tensor<float_t, S6> &prev_delta,
                                 const deconv_params &params,
                                 const bool &has_bias,
                                 const bool &parallelize) {
  const float_t *W = weights.host_pbegin();
  // propagate delta to previous layer
  for_i(parallelize, prev_out.size(), [&](size_t sample) {
    float_t *curr_d_begin       = curr_delta.host_pointer(sample, 0);
    float_t *prev_d_begin       = prev_delta.host_pointer(sample, 0);
    float_t *dW                 = weights_grads.host_pointer(sample, 0);
    float_t *db                 = bias_grads.host_pointer(sample, 0);
    const float_t *prev_o_begin = prev_out.host_pointer(sample, 0);
    for (size_t inc = 0; inc < params.in.depth_; inc++) {
      for (size_t outc = 0; outc < params.out.depth_; outc++) {
        if (!params.tbl.is_connected(outc, inc)) continue;

        size_t idx        = 0;
        idx               = params.in.depth_ * outc + inc;
        idx               = params.weight.get_index(0, 0, idx);
        const float_t *pw = &W[idx];

        idx                       = params.out_unpadded.get_index(0, 0, outc);
        const float_t *pdelta_src = &curr_d_begin[idx];

        idx                 = params.in.get_index(0, 0, inc);
        float_t *pdelta_dst = &prev_d_begin[idx];

        for (size_t y = 0; y < params.in.height_; y++) {
          for (size_t x = 0; x < params.in.width_; x++) {
            const float_t *ppw = pw;

            float_t *ppdelta_dst = pdelta_dst + y * params.in.width_ + x;
            float_t sum{0};

            for (size_t wy = 0; wy < params.weight.height_; wy++) {
              for (size_t wx = 0; wx < params.weight.width_; wx++) {
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
    for (size_t inc = 0; inc < params.in.depth_; inc++) {
      for (size_t outc = 0; outc < params.out.depth_; outc++) {
        if (!params.tbl.is_connected(outc, inc)) continue;

        for (size_t wy = 0; wy < params.weight.height_; wy++) {
          for (size_t wx = 0; wx < params.weight.width_; wx++) {
            float_t dst{0};

            size_t idx           = 0;
            idx                  = params.in.get_index(0, 0, inc);
            const float_t *prevo = &prev_o_begin[idx];

            idx                  = params.out.get_index(wx, wy, outc);
            const float_t *delta = &curr_d_begin[idx];

            for (size_t y = 0; y < params.in.height_; y++) {
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
    if (has_bias) {
      // vec_t& db = *in_grad[2];

      for (size_t outc = 0; outc < params.out.depth_; outc++) {
        size_t idx           = params.out.get_index(0, 0, outc);
        const float_t *delta = &curr_d_begin[idx];
        // TODO: name
        const float_t *deltaa = delta + params.out.width_ * params.out.height_;
        db[outc] += std::accumulate(delta, deltaa, float_t{0});
      }
    }
  });
}

}  // namespace kernels
}  // namespace tiny_dnn
