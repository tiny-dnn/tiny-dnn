// Copyright (c) 2016, Taiga Nomi, Edgar Riba. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#pragma once

namespace tiny_dnn {
namespace core {
namespace kernels {

inline void tiny_maxpool_kernel(const tensor_t& in_data,
                                tensor_t&       out_data,
                                std::vector<std::vector<cnn_size_t>>& max_idx,
                                const std::vector<std::vector<cnn_size_t>>& out2in,
                                const bool layer_parallelize) {

    for_i(layer_parallelize, in_data.size(), [&](int sample) {
        const vec_t& in = in_data[sample];
        vec_t& a = out_data[sample];
        std::vector<cnn_size_t>& max = max_idx[sample];

        for (cnn_size_t i = 0; i < out2in.size(); i++) {
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

inline void tiny_maxpool_back_kernel(tensor_t& prev_delta,
                                     const tensor_t&  curr_delta,
                                     std::vector<std::vector<cnn_size_t>>& max_idx,
                                     const std::vector<cnn_size_t>& in2out,
                                     const bool layer_parallelize) {

    for_i(layer_parallelize, prev_delta.size(), [&](int sample) {
        vec_t& prev       = prev_delta[sample];
        const vec_t& curr = curr_delta[sample];
        const std::vector<cnn_size_t>& max = max_idx[sample];

        for (cnn_size_t i = 0; i < in2out.size(); i++) {
            cnn_size_t outi = in2out[i];
            prev[i] = (max[outi] == static_cast<cnn_size_t>(i)) ?
                       curr[outi] : float_t(0);
        }
    });
    
}

}  // namespace kernels
}  // namespace core
}  // namespace tiny_dnn
