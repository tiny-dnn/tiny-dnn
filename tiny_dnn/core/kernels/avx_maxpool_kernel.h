// Copyright (c) 2016, Taiga Nomi, Edgar Riba. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#pragma once
#include "tiny_dnn/core/kernels/tiny_maxpool_kernel.h"

namespace tiny_dnn {
namespace core {
namespace kernels {

inline void avx_maxpool_kernel(
    const tensor_t& in_data, tensor_t& out_data,
    std::vector<std::vector<cnn_size_t>>& max_idx,
    const std::vector<std::vector<cnn_size_t>>& out2in,
    const bool layer_parallelize) {
  tiny_maxpool_kernel(in_data, out_data, max_idx, out2in, layer_parallelize);
}

inline void avx_maxpool_back_kernel(
    tensor_t& prev_delta, const tensor_t& curr_delta,
    std::vector<std::vector<cnn_size_t>>& max_idx,
    const std::vector<cnn_size_t>& in2out, const bool layer_parallelize) {
  tiny_maxpool_back_kernel(prev_delta, curr_delta, max_idx, in2out,
                           layer_parallelize);
}

}  // namespace kernels
}  // namespace core
}  // namespace tiny_dnn
