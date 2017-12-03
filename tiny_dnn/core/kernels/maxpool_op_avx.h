/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <vector>

#include "tiny_dnn/core/kernels/maxpool_op_internal.h"

namespace tiny_dnn {
namespace kernels {

inline void maxpool_op_avx(const tensor_t &in_data,
                           tensor_t &out_data,
                           std::vector<std::vector<size_t>> &max_idx,
                           const std::vector<std::vector<size_t>> &out2in,
                           const bool layer_parallelize) {
  maxpool_op_internal(in_data, out_data, max_idx, out2in, layer_parallelize);
}

inline void maxpool_grad_op_avx(tensor_t &prev_delta,
                                const tensor_t &curr_delta,
                                std::vector<std::vector<size_t>> &max_idx,
                                const std::vector<size_t> &in2out,
                                const bool layer_parallelize) {
  maxpool_grad_op_internal(prev_delta, curr_delta, max_idx, in2out,
                           layer_parallelize);
}

}  // namespace kernels
}  // namespace tiny_dnn
