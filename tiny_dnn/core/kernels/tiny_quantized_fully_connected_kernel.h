/*
    Copyright (c) 2016, Taiga Nomi, Edgar Riba
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include "tiny_dnn/core/params/fully_params.h"
#include "tiny_dnn/core/kernels/tiny_quantization_kernel.h"
#include "tiny_dnn/core/kernels/tiny_quantized_matmul_kernel.h"

namespace tiny_dnn {
namespace core {
namespace kernels {

inline void tiny_quantized_fully_connected_kernel(const fully_params& params,
                                                  const vec_t&        in,
                                                  const vec_t&        W,
                                                  const vec_t&        b,
                                                  vec_t&              a,
                                                  const bool          layer_parallelize) {
    // input quantization
    float_t min_input(in[0]);
    float_t max_input(in[0]);
    for (serial_size_t c = 0; c < params.in_size_; c++) {
        min_input = std::min(min_input, in[c]);
        max_input = std::max(max_input, in[c]);
    }
    std::vector<uint8_t> in_quantized =
        float_tensor_to_quantized<uint8_t>(in, min_input, max_input);
    // filter quantization
    float_t min_filter(W[0]);
    float_t max_filter(W[0]);
    for (serial_size_t c = 0; c < W.size(); c++) {
        min_filter = std::min(min_filter, W[c]);
        max_filter = std::max(max_filter, W[c]);
    }
    if (min_filter == max_filter) {
      max_filter = W[0] + 1e-3f;
      min_filter = W[0] - 1e-3f;
    }
    std::vector<uint8_t> W_quantized =
        float_tensor_to_quantized<uint8_t>(W, min_filter, max_filter);
    // output range
    float_t min_output_value;
    float_t max_output_value;
    quantization_range_for_multiplication<uint8_t, uint8_t, int32_t>(
        min_input, max_input, min_filter, max_filter, &min_output_value,
        &max_output_value);
    // bias quantization
    float_t min_bias(0);
    float_t max_bias(0);
    std::vector<uint8_t> bias_quantized;
    if (params.has_bias_) {
        for (serial_size_t inc = 0; inc < b.size(); inc++) {
            min_bias = std::min(min_bias, b[inc]);
            max_bias = std::max(max_bias, b[inc]);
        }
        if (min_bias == max_bias) {
          max_bias = b[0] + 1e-3f;
          min_bias = b[0] - 1e-3f;
        }
        bias_quantized =
            float_tensor_to_quantized<uint8_t>(b, min_bias, max_bias);
    }
    min_output_value += min_bias;
    max_output_value += max_bias;

    std::vector<int32_t> a_quantized(a.size(), static_cast<int32_t>(0));

    // calculating offset
    const int32_t offset_input =
        float_to_quantized_unclamped<uint8_t>(0.0f, min_input, max_input);
    const int32_t offset_filter =
        float_to_quantized_unclamped<uint8_t>(0.0f, min_filter, max_filter);
    const int32_t zero_in_total_space =
        float_to_quantized<int32_t>(0.0f, min_output_value, max_output_value);

    const int32_t offset_output = 0;
    const int32_t mult_output = 1;
    const int32_t shift_output = 0;

    bool use_gemm = false;
    if (use_gemm) {
        std::vector<size_t> shape{params.in_size_, 1, params.out_size_, params.in_size_};
        tiny_quantized_matmul(in_quantized,
                              W_quantized,
                              a_quantized,
                              shape,
                              offset_input,
                              offset_filter,
                              offset_output,
                              mult_output,
                              shift_output);
        if (params.has_bias_) {
            for_i(layer_parallelize, params.out_size_, [&](int i) {
            a[i] += b[i];
        });
    }
    } else {
        for_i(layer_parallelize, params.out_size_, [&](int i) {
            for (serial_size_t c = 0; c < params.in_size_; c++) {
                a_quantized[i] += static_cast<int32_t>(W_quantized[c * params.out_size_ + i] - offset_filter) *
                 static_cast<int32_t>(in_quantized[c] - offset_input);
            }
            if (params.has_bias_) {
                a_quantized[i] += (bias_quantized[i] - zero_in_total_space);
            }
        });
    }

    float_t min_output_requantized;
    float_t max_output_requantized;
    std::vector<uint8_t> a_requantized(a_quantized.size(), static_cast<uint8_t>(0));

    // Requantize from 32bits to 8 bits for next layer
    quantize_down_and_shrink_range<int32_t, uint8_t>(a_quantized, min_output_value, max_output_value,
    &min_output_requantized, &max_output_requantized, &a_requantized);

    // dequantize to flaot, this could be removed within concatenated quantized network
    a = quantized_tensor_to_float<uint8_t>(a_requantized, min_output_requantized, max_output_requantized);
}

inline void tiny_quantized_fully_connected_back_kernel(const fully_params& params,
                                                       const vec_t& prev_out,
                                                       const vec_t& W,
                                                       vec_t&       dW,
                                                       vec_t&       prev_delta,
                                                       vec_t&       curr_delta,
                                                       vec_t&       db,
                                                       const bool   layer_parallelize) {
    // previous output quantization
    float_t min_prev_out(prev_out[0]);
    float_t max_prev_out(prev_out[0]);
    for (serial_size_t inc = 0; inc < prev_out.size(); inc++) {
        min_prev_out = std::min(min_prev_out, prev_out[inc]);
        max_prev_out = std::max(min_prev_out, prev_out[inc]);
    }
    std::vector<uint8_t> prev_out_quantized =
        float_tensor_to_quantized<uint8_t>(prev_out, min_prev_out, max_prev_out);

    // filter quantization
    float_t min_filter(W[0]);
    float_t max_filter(W[0]);
    for (serial_size_t c = 0; c < W.size(); c++) {
        min_filter = std::min(min_filter, W[c]);
        max_filter = std::max(max_filter, W[c]);
    }
    if (min_filter == max_filter) {
      max_filter = W[0] + 1e-3f;
      min_filter = W[0] - 1e-3f;
    }
    std::vector<uint8_t> W_quantized =
        float_tensor_to_quantized<uint8_t>(W, min_filter, max_filter);

    // current delta quantization
    float_t min_curr_delta(curr_delta[0]);
    float_t max_curr_delta(curr_delta[0]);
    for (serial_size_t inc = 0; inc < curr_delta.size(); inc++) {
            min_curr_delta = std::min(min_curr_delta, curr_delta[inc]);
            max_curr_delta = std::max(max_curr_delta, curr_delta[inc]);
    }
    std::vector<uint8_t> curr_delta_quantized =
        float_tensor_to_quantized<uint8_t>(curr_delta, min_curr_delta, max_curr_delta);

    // output range for previous delta
    float_t min_prev_delta_value;
    float_t max_prev_delta_value;
    quantization_range_for_multiplication<uint8_t, uint8_t, int32_t>(
        min_curr_delta, max_curr_delta, min_filter, max_filter, &min_prev_delta_value,
        &max_prev_delta_value);

    std::vector<int32_t> prev_delta_quantized(prev_delta.size(), static_cast<int32_t>(0));

    // output range for dW
    float_t min_dW_value;
    float_t max_dW_value;
    quantization_range_for_multiplication<uint8_t, uint8_t, int32_t>(
        min_curr_delta, max_curr_delta, min_prev_out, max_prev_out, &min_dW_value,
        &max_dW_value);

    std::vector<int32_t> dW_quantized(dW.size(), static_cast<int32_t>(0));

    // calculating offset
    const int32_t offset_prev_out =
        float_to_quantized_unclamped<uint8_t>(0.0f, min_prev_out, max_prev_out);
    const int32_t offset_filter =
        float_to_quantized_unclamped<uint8_t>(0.0f, min_filter, max_filter);
    const int32_t offset_curr_delta =
        float_to_quantized_unclamped<uint8_t>(0.0f, min_curr_delta, max_curr_delta);
    //const int32_t zero_in_prev_delta =
    //    float_to_quantized<int32_t>(0.0f, min_prev_delta_value, max_prev_delta_value);

    for (serial_size_t c = 0; c < params.in_size_; c++) {
        // propagate delta to previous layer
        // prev_delta[c] += current_delta[r] * W_[c * out_size_ + r]
        for (serial_size_t io = 0; io < params.out_size_; io++) {
            prev_delta_quantized[c] += (static_cast<int32_t>(curr_delta_quantized[io]) - offset_curr_delta)
                                       * (static_cast<int32_t>(W_quantized[c * params.out_size_ + io]) - offset_filter);
        }
    }

    float_t min_prev_delta_requantized;
    float_t max_prev_delta_requantized;
    std::vector<uint8_t> prev_delta_requantized(prev_delta_quantized.size(), static_cast<uint8_t>(0));

    // Requantize from 32bits to 8 bits for next layer
    quantize_down_and_shrink_range<int32_t, uint8_t>(prev_delta_quantized, min_prev_delta_value, max_prev_delta_value,
    &min_prev_delta_requantized, &max_prev_delta_requantized, &prev_delta_requantized);

    // dequantize to flaot, this could be removed within concatenated quantized network
    prev_delta = quantized_tensor_to_float<uint8_t>(prev_delta_requantized, min_prev_delta_requantized, max_prev_delta_requantized);

    for_(layer_parallelize, 0, size_t(params.out_size_), [&](const blocked_range& r) {
        // accumulate weight-step using delta
        // dW[c * out_size + i] += current_delta[i] * prev_out[c]
        for (serial_size_t c = 0; c < params.in_size_; c++) {
            for (serial_size_t io = 0; io < params.out_size_; io++) {
                dW_quantized[c * params.out_size_ + io] += (static_cast<int32_t>(curr_delta_quantized[io]) - offset_curr_delta)
                                                   * (static_cast<int32_t>(prev_out_quantized[c]) - offset_prev_out);
            }
        }

        if (params.has_bias_) {
            // vec_t& db = *in_grad[2];
            for (int i = r.begin(); i < r.end(); i++) {
                db[i] += curr_delta[i];
            }
        }
    });

    float_t min_dW_requantized;
    float_t max_dW_requantized;
    std::vector<uint8_t> dW_requantized(dW_quantized.size(), static_cast<uint8_t>(0));

    // requantize from 32bits to 8 bits for next layer
    quantize_down_and_shrink_range<int32_t, uint8_t>(dW_quantized, min_dW_value, max_dW_value,
    &min_dW_requantized, &max_dW_requantized, &dW_requantized);

    // dequantize to flaot, this could be removed within concatenated quantized network
    dW = quantized_tensor_to_float<uint8_t>(dW_requantized, min_dW_requantized, max_dW_requantized);
}

inline void tiny_quantized_fully_connected_kernel(const fully_params& params,
                                                  const vec_t&        in,
                                                  const vec_t&        W,
                                                  const vec_t&        b,
                                                  const vec_t&        in_r,
                                                  const vec_t&        W_r,
                                                  const vec_t&        b_r,
                                                  vec_t&              a,
                                                  vec_t&              a_r,
                                                  const bool          layer_parallelize) {
    // filter range
    float_t min_filter(W_r[0]);
    float_t max_filter(W_r[1]);
    if (min_filter == max_filter) {
      max_filter = W_r[1] + 1e-3f;
      min_filter = W_r[0] - 1e-3f;
    }
    // bias range
    float_t min_bias(b_r[0]);
    float_t max_bias(b_r[1]);
    if (params.has_bias_) {
        if (min_bias == max_bias) {
          max_bias = b_r[1] + 1e-3f;
          min_bias = b_r[0] - 1e-3f;
        }
    }
    // output range
    float_t min_output_value;
    float_t max_output_value;
    quantization_range_for_multiplication<uint8_t, uint8_t, int32_t>(
        in_r[0], in_r[1], min_filter, max_filter, &min_output_value,
        &max_output_value);
    // data type restore
    std::vector<uint8_t> in_quantized, W_quantized, bias_quantized;
    for (size_t i = 0; i < in.size(); i++) {
       in_quantized.push_back(static_cast<uint8_t>(in[i]));
    }
    for (size_t i = 0; i < W.size(); i++) {
        W_quantized.push_back(static_cast<uint8_t>(W[i]));
    }
    for (size_t i = 0; i < b.size(); i++) {
        bias_quantized.push_back(static_cast<uint8_t>(b[i]));
    }
    min_output_value += min_bias;
    max_output_value += max_bias;

    std::vector<int32_t> a_quantized(a.size(), static_cast<int32_t>(0));

    // calculating offset
    const int32_t offset_input =
        float_to_quantized_unclamped<uint8_t>(0.0f, in_r[0], in_r[1]);
    const int32_t offset_filter =
        float_to_quantized_unclamped<uint8_t>(0.0f, min_filter, max_filter);
    const int32_t zero_in_total_space =
        float_to_quantized<int32_t>(0.0f, min_output_value, max_output_value);

    const int32_t offset_output = 0;
    const int32_t mult_output = 1;
    const int32_t shift_output = 0;

    bool use_gemm = false;
    if (use_gemm) {
        std::vector<size_t> shape{params.in_size_, 1, params.out_size_, params.in_size_};
        tiny_quantized_matmul(in_quantized,
                              W_quantized,
                              a_quantized,
                              shape,
                              offset_input,
                              offset_filter,
                              offset_output,
                              mult_output,
                              shift_output);
        if (params.has_bias_) {
            for_i(layer_parallelize, params.out_size_, [&](int i) {
            a[i] += b[i];
        });
    }
    } else {
        for_i(layer_parallelize, params.out_size_, [&](int i) {
            for (serial_size_t c = 0; c < params.in_size_; c++) {
                a_quantized[i] += static_cast<int32_t>(W_quantized[c * params.out_size_ + i] - offset_filter) *
                 static_cast<int32_t>(in_quantized[c] - offset_input);
            }
            if (params.has_bias_) {
                a_quantized[i] += (bias_quantized[i] - zero_in_total_space);
            }
        });
    }

    float_t min_output_requantized;
    float_t max_output_requantized;
    std::vector<uint8_t> a_requantized(a_quantized.size(), static_cast<uint8_t>(0));

    // Requantize from 32bits to 8 bits for next layer
    quantize_down_and_shrink_range<int32_t, uint8_t>(a_quantized, min_output_value, max_output_value,
        &min_output_requantized, &max_output_requantized, &a_requantized);
    // store directly in float datatype
    for (size_t i = 0; i < a_requantized.size(); i++) {
        a[i] = static_cast<float>(a_requantized[i]);
    }
    a_r[0] = min_output_requantized;
    a_r[1] = max_output_requantized;
}

}  // namespace kernels
}  // namespace core
}  // namespace tiny_dnn
