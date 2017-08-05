/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <algorithm>
#include <vector>

#include "tiny_dnn/core/framework/tensor.h"
#include "tiny_dnn/core/kernels/tiny_quantization_kernel.h"
#include "tiny_dnn/core/params/conv_params.h"

namespace tiny_dnn {
namespace core {
namespace kernels {

template <typename S1, typename S2, typename S3, typename S4>
inline void tiny_quantized_conv2d_kernel(const conv_params &params,
                                         const Tensor<float_t, S1> &in,
                                         const Tensor<float_t, S2> &W,
                                         const Tensor<float_t, S3> &bias,
                                         Tensor<float_t, S4> &a,
                                         const bool layer_parallelize) {
  float_t min_input, max_input, min_filter, max_filter, min_bias, max_bias;

  auto in_quantized   = quantize_tensor<uint8_t>(in, min_input, max_input);
  auto W_quantized    = quantize_tensor<uint8_t>(W, min_filter, max_filter);
  auto bias_quantized = params.has_bias
                          ? quantize_tensor<uint8_t>(bias, min_bias, max_bias)
                          : Tensor<uint8_t>();

  // output range
  float_t min_output_value, max_output_value;
  quantization_range_for_multiplication<uint8_t, uint8_t, int32_t>(
    min_input, max_input, min_filter, max_filter, &min_output_value,
    &max_output_value);

  Tensor<int32_t> a_quantized({a.size()});
  a_quantized.fill(int32_t(0));
  // calculating offset
  const int32_t offset_input = int64_to_int32(
    float_to_quantized_unclamped<uint8_t>(0.0f, min_input, max_input));
  const int32_t offset_filter = int64_to_int32(
    float_to_quantized_unclamped<uint8_t>(0.0f, min_filter, max_filter));
  const int32_t zero_in_total_space = int64_to_int32(
    float_to_quantized<int32_t>(0.0f, min_output_value, max_output_value));

  for_i(layer_parallelize, params.out.depth_, [&](size_t o) {
    for (size_t inc = 0; inc < params.in.depth_; inc++) {
      if (!params.tbl.is_connected(o, inc)) continue;

      size_t idx        = 0;
      idx               = params.in.depth_ * o + inc;
      idx               = params.weight.get_index(0, 0, idx);
      const uint8_t *pw = W_quantized.host_pointer(idx);

      idx               = params.in_padded.get_index(0, 0, inc);
      const uint8_t *pi = in_quantized.host_pointer(idx);

      idx                   = params.out.get_index(0, 0, o);
      int32_t *pa_quantized = a_quantized.host_pointer(idx);

      for (size_t y = 0; y < params.out.height_; y++) {
        for (size_t x = 0; x < params.out.width_; x++) {
          const uint8_t *ppw = pw;
          const uint8_t *ppi = pi +
                               params.in_padded.width_ * (y * params.h_stride) +
                               x * params.w_stride;
          int32_t sum = 0;

          // should be optimized for small kernel(3x3,5x5)
          for (size_t wy = 0; wy < params.weight.height_; wy++) {
            for (size_t wx = 0; wx < params.weight.width_; wx++) {
              idx = wy * params.in_padded.width_ + wx;
              sum += (static_cast<int32_t>(*ppw++) - offset_filter) *
                     (static_cast<int32_t>(ppi[idx]) - offset_input);
            }
          }
          pa_quantized[y * params.out.width_ + x] += sum;
        }
      }
    }
    if (params.has_bias) {
      int32_t *pa_quantized =
        a_quantized.host_pointer(params.out.get_index(0, 0, o));
      int32_t *paa_quantized =
        pa_quantized + params.out.width_ * params.out.height_;
      std::for_each(pa_quantized, paa_quantized, [&](int32_t &f) {
        f += (bias_quantized.host_at(o) - zero_in_total_space);
      });
    }
  });

  float_t min_output_requantized;
  float_t max_output_requantized;
  Tensor<uint8_t> a_requantized({a_quantized.size()});
  a_requantized.fill(uint8_t(0));

  // Requantize from 32bits to 8 bits for next layer
  quantize_down_and_shrink_range<int32_t, uint8_t>(
    a_quantized, min_output_value, max_output_value, &min_output_requantized,
    &max_output_requantized, &a_requantized);

  // TODO(Randl): different types
  // dequantize to float, this could be removed within concatenated quantized
  // network
  a = quantized_tensor_to_float<uint8_t>(a_requantized, min_output_requantized,
                                         max_output_requantized);
}

// TODO(Randl): Refactor
template <typename S1,
          typename S2,
          typename S3,
          typename S4,
          typename S5,
          typename S6>
inline void tiny_quantized_conv2d_back_kernel(
  const conv_params &params,
  const Tensor<float_t, S1> &prev_out,
  const Tensor<float_t, S2> &W,
  Tensor<float_t, S3> &dW,
  Tensor<float_t, S4> &db,
  Tensor<float_t, S5> &curr_delta,
  Tensor<float_t, S6> *prev_delta) {
  float_t min_prev_out, max_prev_out, min_filter, max_filter, min_curr_delta,
    max_curr_delta;
  auto prev_out_quantized =
    quantize_tensor<uint8_t>(prev_out, min_prev_out, max_prev_out);
  auto W_quantized = quantize_tensor<uint8_t>(W, min_filter, max_filter);
  auto curr_delta_quantized =
    quantize_tensor<uint8_t>(curr_delta, min_curr_delta, max_curr_delta);

  // output range for previous delta
  float_t min_prev_delta_value, max_prev_delta_value;
  quantization_range_for_multiplication<uint8_t, uint8_t, int32_t>(
    min_curr_delta, max_curr_delta, min_filter, max_filter,
    &min_prev_delta_value, &max_prev_delta_value);

  Tensor<int32_t> prev_delta_quantized({prev_delta->size()});
  prev_delta_quantized.fill(int32_t(0));

  // output range for dW
  float_t min_dW_value, max_dW_value;
  quantization_range_for_multiplication<uint8_t, uint8_t, int32_t>(
    min_curr_delta, max_curr_delta, min_prev_out, max_prev_out, &min_dW_value,
    &max_dW_value);

  Tensor<int32_t> dW_quantized({dW.size()});
  prev_delta_quantized.fill(int32_t(0));

  // calculating offset
  const int32_t offset_prev_out = int64_to_int32(
    float_to_quantized_unclamped<uint8_t>(0.0f, min_prev_out, max_prev_out));
  const int32_t offset_filter = int64_to_int32(
    float_to_quantized_unclamped<uint8_t>(0.0f, min_filter, max_filter));
  const int32_t offset_curr_delta =
    int64_to_int32(float_to_quantized_unclamped<uint8_t>(0.0f, min_curr_delta,
                                                         max_curr_delta));

  // propagate delta to previous layer
  for_i(params.in.depth_, [&](size_t inc) {
    for (size_t outc = 0; outc < params.out.depth_; outc++) {
      if (!params.tbl.is_connected(outc, inc)) continue;

      size_t idx        = 0;
      idx               = params.in.depth_ * outc + inc;
      idx               = params.weight.get_index(0, 0, idx);
      const uint8_t *pw = W_quantized.host_pointer(idx);

      idx                       = params.out.get_index(0, 0, outc);
      const uint8_t *pdelta_src = curr_delta_quantized.host_pointer(idx);

      idx                           = params.in_padded.get_index(0, 0, inc);
      int32_t *pdelta_quantized_dst = prev_delta_quantized.host_pointer(idx);

      for (size_t y = 0; y < params.out.height_; y++) {
        for (size_t x = 0; x < params.out.width_; x++) {
          const uint8_t *ppw = pw;

          idx                       = y * params.out.width_ + x;
          const uint8_t ppdelta_src = pdelta_src[idx];

          int32_t *ppdelta_quantized_dst =
            pdelta_quantized_dst +
            y * params.h_stride * params.in_padded.width_ + x * params.w_stride;

          for (size_t wy = 0; wy < params.weight.height_; wy++) {   // NOLINT
            for (size_t wx = 0; wx < params.weight.width_; wx++) {  // NOLINT
              idx = wy * params.in_padded.width_ + wx;
              ppdelta_quantized_dst[idx] +=
                (static_cast<int32_t>(*ppw++) - offset_filter) *
                (static_cast<int32_t>(ppdelta_src) - offset_curr_delta);
            }
          }
        }
      }
    }
  });

  float_t min_prev_delta_requantized;
  float_t max_prev_delta_requantized;
  Tensor<uint8_t> prev_delta_requantized({prev_delta_quantized.size()});
  prev_delta_requantized.fill(uint8_t(0));

  // Requantize from 32bits to 8 bits for next layer
  quantize_down_and_shrink_range<int32_t, uint8_t>(
    prev_delta_quantized, min_prev_delta_value, max_prev_delta_value,
    &min_prev_delta_requantized, &max_prev_delta_requantized,
    &prev_delta_requantized);

  // dequantize to float, this could be removed within concatenated quantized
  // network
  Tensor<> prev_delta_vec = quantized_tensor_to_float<uint8_t>(
    prev_delta_requantized, min_prev_delta_requantized,
    max_prev_delta_requantized);
  prev_delta = &prev_delta_vec;

  // Accumulate dw
  for_i(params.in.depth_, [&](size_t inc) {
    for (size_t outc = 0; outc < params.out.depth_; outc++) {
      if (!params.tbl.is_connected(outc, inc)) continue;

      for (size_t wy = 0; wy < params.weight.height_; wy++) {
        for (size_t wx = 0; wx < params.weight.width_; wx++) {
          int32_t dst = int32_t(0);

          size_t idx           = 0;
          idx                  = params.in_padded.get_index(wx, wy, inc);
          const uint8_t *prevo = prev_out_quantized.host_pointer(idx);

          idx                  = params.out.get_index(0, 0, outc);
          const uint8_t *delta = curr_delta_quantized.host_pointer(idx);

          for (size_t y = 0; y < params.out.height_; y++) {
            for (size_t x = 0; x < params.out.width_; x++) {
              dst +=
                (static_cast<int32_t>(
                   *(prevo + y * params.in_padded.width_ + x)) -
                 offset_prev_out) *
                (static_cast<int32_t>(*(delta + y * params.out.width_ + x)) -
                 offset_curr_delta);
            }
          }

          idx = params.in.depth_ * outc + inc;
          dW_quantized.host_at(params.weight.get_index(wx, wy, idx)) += dst;
        }
      }
    }
  });

  float_t min_dW_requantized;
  float_t max_dW_requantized;
  Tensor<uint8_t> dW_requantized({dW_quantized.size()});
  dW_requantized.fill(uint8_t(0));

  // requantize from 32bits to 8 bits for next layer
  quantize_down_and_shrink_range<int32_t, uint8_t>(
    dW_quantized, min_dW_value, max_dW_value, &min_dW_requantized,
    &max_dW_requantized, &dW_requantized);

  // dequantize to float, this could be removed within concatenated quantized
  // network
  dW = quantized_tensor_to_float<uint8_t>(dW_requantized, min_dW_requantized,
                                          max_dW_requantized);

  // Accumulate db
  if (params.has_bias) {
    // vec_t& db = *in_grad[2];

    for (size_t outc = 0; outc < params.out.depth_; outc++) {
      size_t idx            = params.out.get_index(0, 0, outc);
      const float_t *delta  = curr_delta.host_pointer(idx);
      const float_t *deltaa = delta + params.out.width_ * params.out.height_;
      db.host_at(outc) += std::accumulate(delta, deltaa, float_t(0.0));
    }
  }
}

// TODO(Randl): ranges via std::pair
template <typename S1,
          typename S2,
          typename S3,
          typename S4,
          typename S5,
          typename S6,
          typename S7,
          typename S8>
inline void tiny_quantized_conv2d_kernel(const conv_params &params,
                                         const Tensor<float_t, S1> &in,
                                         const Tensor<float_t, S2> &W,
                                         const Tensor<float_t, S3> &bias,
                                         const Tensor<float_t, S4> &in_r,
                                         const Tensor<float_t, S5> &W_r,
                                         const Tensor<float_t, S6> &b_r,
                                         Tensor<float_t, S7> &a,
                                         Tensor<float_t, S8> &a_r,
                                         const bool layer_parallelize) {
  // filter range
  float_t min_filter(W_r.host_at(0));
  float_t max_filter(W_r.host_at(1));
  if (min_filter == max_filter) {
    max_filter += 1e-3f;
    min_filter -= 1e-3f;
  }
  // bias range
  float_t min_bias(b_r.host_at(0));
  float_t max_bias(b_r.host_at(1));
  if (params.has_bias) {
    if (min_bias == max_bias) {
      max_bias += 1e-3f;
      min_bias -= 1e-3f;
    }
  }
  // output range
  float_t min_output_value;
  float_t max_output_value;
  quantization_range_for_multiplication<uint8_t, uint8_t, int32_t>(
    in_r.host_at(0), in_r.host_at(1), min_filter, max_filter, &min_output_value,
    &max_output_value);
  // data type restore
  Tensor<uint8_t> in_quantized({in.size()}), W_quantized({W.size()}),
    bias_quantized({bias.size()});
  // TODO(Randl): iterators?
  for (size_t i = 0; i < in.size(); i++) {
    in_quantized.host_at(i) = static_cast<uint8_t>(in.host_at(i));
  }
  for (size_t i = 0; i < W.size(); i++) {
    W_quantized.host_at(i) = static_cast<uint8_t>(W.host_at(i));
  }
  for (size_t i = 0; i < bias.size(); i++) {
    bias_quantized.host_at(i) = static_cast<uint8_t>(bias.host_at(i));
  }

  Tensor<int32_t> a_quantized({a.size()});
  a_quantized.fill(int32_t(0));

  // calculating offset
  const int32_t offset_input =
    int64_to_int32(float_to_quantized_unclamped<uint8_t>(0.0f, in_r.host_at(0),
                                                         in_r.host_at(1)));
  const int32_t offset_filter = int64_to_int32(
    float_to_quantized_unclamped<uint8_t>(0.0f, min_filter, max_filter));
  const int32_t zero_in_total_space = int64_to_int32(
    float_to_quantized<int32_t>(0.0f, min_output_value, max_output_value));

  for_i(layer_parallelize, params.out.depth_, [&](size_t o) {
    for (size_t inc = 0; inc < params.in.depth_; inc++) {
      if (!params.tbl.is_connected(o, inc)) continue;

      size_t idx        = 0;
      idx               = params.in.depth_ * o + inc;
      idx               = params.weight.get_index(0, 0, idx);
      const uint8_t *pw = W_quantized.host_pointer(idx);

      idx               = params.in_padded.get_index(0, 0, inc);
      const uint8_t *pi = in_quantized.host_pointer(idx);

      idx                   = params.out.get_index(0, 0, o);
      int32_t *pa_quantized = a_quantized.host_pointer(idx);

      for (size_t y = 0; y < params.out.height_; y++) {
        for (size_t x = 0; x < params.out.width_; x++) {
          const uint8_t *ppw = pw;
          const uint8_t *ppi = pi +
                               params.in_padded.width_ * (y * params.h_stride) +
                               x * params.w_stride;
          int32_t sum = 0;

          // should be optimized for small kernel(3x3,5x5)
          for (size_t wy = 0; wy < params.weight.height_; wy++) {
            for (size_t wx = 0; wx < params.weight.width_; wx++) {
              idx = wy * params.in_padded.width_ + wx;
              sum += (static_cast<int32_t>(*ppw++) - offset_filter) *
                     (static_cast<int32_t>(ppi[idx]) - offset_input);
            }
          }
          pa_quantized[y * params.out.width_ + x] += sum;
        }
      }
    }
    if (params.has_bias) {
      int32_t *pa_quantized =
        a_quantized.host_pointer(params.out.get_index(0, 0, o));
      int32_t *paa_quantized =
        pa_quantized + params.out.width_ * params.out.height_;
      std::for_each(pa_quantized, paa_quantized, [&](int32_t &f) {
        f += static_cast<int32_t>((bias.host_at(o) - zero_in_total_space));
      });
    }
  });

  float_t min_output_requantized;
  float_t max_output_requantized;
  Tensor<uint8_t> a_requantized({a_quantized.size()});
  a_requantized.fill(uint8_t(0));

  // Requantize from 32bits to 8 bits for next layer
  quantize_down_and_shrink_range<int32_t, uint8_t>(
    a_quantized, min_output_value, max_output_value, &min_output_requantized,
    &max_output_requantized, &a_requantized);
  // store directly in float datatype
  for (size_t i = 0; i < a_requantized.size(); i++) {
    a.host_at(i) = static_cast<float_t>(a_requantized.host_at(i));
  }
  a_r.host_at(0) = min_output_requantized;
  a_r.host_at(1) = max_output_requantized;
}

}  // namespace kernels
}  // namespace core
}  // namespace tiny_dnn
