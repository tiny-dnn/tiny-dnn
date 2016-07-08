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

#include "tiny_cnn/core/params/fully_params.h"
#include "tiny_cnn/core/kernels/tiny_quantization_kernel.h"
#include "tiny_cnn/core/kernels/tiny_quantized_matmul_kernel.h"

namespace tiny_cnn {
namespace core {
namespace kernels {

void tiny_quantized_fully_connected_kernel(const fully_params& params,
                                 const vec_t&        in,
                                 const vec_t&        W,
                                 const vec_t&        b,
                                 vec_t&              a,
                                 const bool          layer_parallelize) {
    // input quantization
    float min_input(in[0]);
    float max_input(in[0]);
    for (cnn_size_t c = 0; c < params.in_size_; c++) {
        min_input = std::min(min_input, in[c]);
        max_input = std::max(max_input, in[c]);
    }
    std::vector<uint8_t> in_quantized =
        float_tensor_to_quantized<uint8_t>(in, min_input, max_input);
    // filter quantization
    float min_filter(W[0]);
    float max_filter(W[0]);
    for (cnn_size_t c = 0; c < W.size(); c++) {
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
    float min_output_value;
    float max_output_value;
    quantization_range_for_multiplication<uint8_t, uint8_t, int32_t>(
        min_input, max_input, min_filter, max_filter, &min_output_value,
        &max_output_value);
    // bias quantization
    float min_bias(0);
    float max_bias(0);
    std::vector<uint8_t> bias_quantized;
    if (params.has_bias_) {
        for (cnn_size_t inc = 0; inc < b.size(); inc++) {
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

    const int32_t rounding = (shift_output < 1) ? 0 : (1 << (shift_output - 1));
    const int32_t highest_ = static_cast<int32_t>(highest<uint8_t>());
    const int32_t lowest_ = static_cast<int32_t>(lowest<uint8_t>());

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
            for (cnn_size_t c = 0; c < params.in_size_; c++) {
                a_quantized[i] += static_cast<int32_t>(W_quantized[c * params.out_size_ + i] - offset_filter) *
                 static_cast<int32_t>(in_quantized[c] - offset_input);
            }
            if (params.has_bias_) {
                a_quantized[i] += (bias_quantized[i] - zero_in_total_space);
            }
        });
    }

    float min_output_requantized;
    float max_output_requantized;
    std::vector<uint8_t> a_requantized(a_quantized.size(), static_cast<uint8_t>(0));

    // Requantize from 32bits to 8 bits for next layer
    quantize_down_and_shrink_range<int32_t, uint8_t>(a_quantized, min_output_value, max_output_value,
    &min_output_requantized, &max_output_requantized, &a_requantized);

    // dequantize to flaot, this could be removed within concatenated quantized network
    a = quantized_tensor_to_float<uint8_t>(a_requantized, min_output_requantized, max_output_requantized);
}

}  // namespace kernels
}  // namespace core
}  // namespace tiny_cnn
