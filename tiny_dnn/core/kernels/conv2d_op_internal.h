/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

namespace tiny_dnn {
namespace kernels {

/**
 * Forward pass for convolution with internal backend.
 * @param in_data
 * @param weights
 * @param bias
 * @param out_data
 * @param params
 * @param parallelize
 */
template <typename S1, typename S2, typename S3, typename S4>
inline void conv2d_op_internal(const Tensor<float_t, S1> &in_data,
                               const Tensor<float_t, S2> &weights,
                               const Tensor<float_t, S3> &bias,
                               Tensor<float_t, S4> &out_data,
                               const core::conv_params &params,
                               const bool parallelize) {
  for_(
    parallelize, 0u, in_data.shape()[0],
    [&](const blocked_range &r) {
      size_t out_area           = params.out.area();
      size_t iw          = params.in_padded.width_;
      size_t id          = params.in.depth_;
      size_t ow          = params.out.width_;
      size_t oh          = params.out.height_;
      size_t od          = params.out.depth_;
      size_t kw          = params.weight.width_;
      size_t kh          = params.weight.height_;
      size_t elem_stride = params.w_stride;
      size_t line_stride = iw * params.h_stride;
      //TODO(edgarriba): replace with  tensor accessors
      for (size_t sample = r.begin(); sample < r.end(); sample++) {
        float_t *out_data_begin = out_data.host_pointer(sample, 0);
        const float_t *in_data_begin = in_data.host_pointer(sample, 0);
        const float_t *weight_begin = weights.host_pointer(0, 0);
        for (size_t o = 0; o < od; o++) {
          // TODO(Randl): naming
          auto pa = &out_data_begin[params.out.get_index(0, 0, o)];
          for (size_t inc = 0; inc < id; inc++) {
            if (!params.tbl.is_connected(o, inc)) continue;
            auto pw =
            &weight_begin[params.weight.get_index(0, 0, id * o + inc)];
            auto pin =
            &in_data_begin[params.in_padded.get_index(0, 0, inc)];
            auto pout = pa;
            for (size_t y = 0; y < oh; y++) {
              auto pin_line = pin;
              for (size_t x = 0; x < ow; x++) {
                auto pw_element = pw;
                float_t sum{0};
                // should be optimized for small kernel(3x3,5x5)
                for (size_t wy = 0; wy < kh; wy++) {  // NOLINT
                  auto pin_element = &pin_line[iw * wy];
                  for (serial_size_t wx = 0; wx < kw; wx++) {  // NOLINT
                    sum += *(pw_element++) * *(pin_element++);
                  }
                }
                *(pout++) += sum;
                pin_line = std::next(pin_line, elem_stride);
              }
              pin = std::next(pin, line_stride);
            }
          }
          if (params.has_bias) {
            vectorize::add(bias.host_at(0, o), out_area, pa);
          }
        }
      }
    },
    0);
}

/******************************************************************/
template <typename S1,
          typename S2,
          typename S3,
          typename S4,
          typename S5,
          typename S6>
inline void conv2d_op_internal(const Tensor<float_t, S1> &prev_out,
                               const Tensor<float_t, S2> &weigths,
                               Tensor<float_t, S3> &weights_grads,
                               Tensor<float_t, S4> &bias_grads,
                               Tensor<float_t, S5> &curr_delta,
                               Tensor<float_t, S6> &prev_delta,
                               const core::conv_params &params,
                               const bool parallelize) {
  for_i(parallelize, prev_out.shape()[0], [&](size_t sample) {
    const float_t *prev = prev_out.host_pointer(sample, 0);
    float_t *curr = curr_delta.host_pointer(sample, 0);
    float_t *prev_d = prev_delta.host_pointer(sample, 0);
    const float_t *weight_begin = weigths.host_pointer(0, 0);
    // propagate delta to previous layer
    for (size_t inc = 0; inc < params.in.depth_; inc++) {
      for (size_t outc = 0; outc < params.out.depth_; outc++) {
        if (!params.tbl.is_connected(outc, inc)) continue;

        size_t idx = 0;

        idx = params.in.depth_ * outc + inc;
        idx = params.weight.get_index(0, 0, idx);

        const float_t *pw = &weight_begin[idx];

        idx = params.out.get_index(0, 0, outc);

        const float_t *pdelta_src = &curr[idx];

        idx                 = params.in_padded.get_index(0, 0, inc);
        float_t *pdelta_dst = &prev_d[idx];

        for (size_t y = 0; y < params.out.height_; y++) {
          for (size_t x = 0; x < params.out.width_; x++) {
            const float_t *ppw = pw;

            idx = y * params.out.width_ + x;

            const float_t ppdelta_src = pdelta_src[idx];

            float_t *ppdelta_dst =
              pdelta_dst + y * params.h_stride * params.in_padded.width_ +
              x * params.w_stride;

            for (size_t wy = 0; wy < params.weight.height_; wy++) {   // NOLINT
              for (size_t wx = 0; wx < params.weight.width_; wx++) {  // NOLINT
                idx = wy * params.in_padded.width_ + wx;
                ppdelta_dst[idx] += *ppw++ * ppdelta_src;
              }
            }
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
            idx                  = params.in_padded.get_index(wx, wy, inc);
            const float_t *prevo = &prev[idx];

            idx                  = params.out.get_index(0, 0, outc);
            const float_t *delta = &curr[idx];

            if (params.w_stride > 1) {
              for (size_t y = 0; y < params.out.height_; y++) {
                size_t prevo_idx =
                  y * params.in_padded.width_ * params.h_stride;
                size_t delta_idx = y * params.out.width_;

                for (size_t x = 0; x < params.out.width_; x++) {
                  dst += prevo[prevo_idx + x * params.w_stride] *
                         delta[delta_idx + x];
                }
              }
            } else {
              for (size_t y = 0; y < params.out.height_; y++) {
                dst += vectorize::dot(
                  prevo + y * params.in_padded.width_ * params.h_stride,
                  delta + y * params.out.width_, params.out.width_);
              }
            }

            idx = params.in.depth_ * outc + inc;
            weights_grads.host_at(sample,
                                  params.weight.get_index(wx, wy, idx)) += dst;
          }
        }
      }
    }

    // accumulate db
    if (params.has_bias) {
      const float_t *delta_begin  = curr_delta.host_pointer(sample, 0);
      for (size_t outc = 0; outc < params.out.depth_; outc++) {
        size_t idx     = params.out.get_index(0, 0, outc);
        const float_t *delta  = &delta_begin[idx];
        const float_t *deltaa = delta + params.out.width_ * params.out.height_;
        bias_grads.host_at(sample, outc) +=
          std::accumulate(delta, deltaa, float_t{0});
      }
    }
  });
}

}  // namespace kernels
}  // namespace tiny_dnn
