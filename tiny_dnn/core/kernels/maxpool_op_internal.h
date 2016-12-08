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

namespace tiny_dnn {
namespace kernels {

inline void
maxpool_op_internal(const tensor_t& in_data,
                    tensor_t&       out_data,
                    std::vector<std::vector<serial_size_t>>& max_idx,
                    const std::vector<std::vector<serial_size_t>>& out2in,
                    const bool layer_parallelize) {
    for_i(layer_parallelize, in_data.size(), [&](int sample) {
        const vec_t& in = in_data[sample];
        vec_t& a = out_data[sample];
        std::vector<serial_size_t>& max = max_idx[sample];

        for (serial_size_t i = 0; i < out2in.size(); i++) {
            const auto& in_index = out2in[i];
            float_t max_value = std::numeric_limits<float_t>::lowest();

            for (auto j : in_index) {
                if (in[j] > max_value) {
                    max_value = in[j];
                    max[i] = j;
                }
            }
            a[i] = max_value;
        }
    });
}

inline void
maxpool_grad_op_internal(tensor_t& prev_delta,
                         const tensor_t&  curr_delta,
                         std::vector<std::vector<serial_size_t>>& max_idx,
                         const std::vector<serial_size_t>& in2out,
                         const bool layer_parallelize) {
    for_i(layer_parallelize, prev_delta.size(), [&](int sample) {
        vec_t& prev       = prev_delta[sample];
        const vec_t& curr = curr_delta[sample];
        const std::vector<serial_size_t>& max = max_idx[sample];

        for (serial_size_t i = 0; i < in2out.size(); i++) {
            serial_size_t outi = in2out[i];
            prev[i] = (max[outi] == static_cast<serial_size_t>(i)) ?
                       curr[outi] : float_t(0);
        }
    }); 
}

}  // namespace kernels
}  // namespace tiny_dnn
