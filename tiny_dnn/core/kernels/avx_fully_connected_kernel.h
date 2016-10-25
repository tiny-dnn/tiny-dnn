// Copyright (c) 2016, Taiga Nomi, Edgar Riba. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#pragma once

#include "tiny_dnn/core/params/fully_params.h"
#include "tiny_dnn/core/kernels/tiny_fully_connected_kernel.h"

namespace tiny_dnn {
namespace core {
namespace kernels {

inline void avx_fully_connected_kernel(const fully_params& params,
                                       const tensor_t& in,
                                       const vec_t&    W,
                                       const vec_t&    b,
                                       tensor_t&       a,
                                       const bool      layer_parallelize) {
    tiny_fully_connected_kernel(params, in, W, b, a, layer_parallelize);
}

inline void avx_fully_connected_back_kernel(const fully_params& params,
                                            const tensor_t& prev_out,
                                            const vec_t&    W,
                                            tensor_t&       dW,
                                            tensor_t&       prev_delta,
                                            tensor_t&       curr_delta,
                                            tensor_t&       db,
                                            const bool      layer_parallelize) {
    tiny_fully_connected_back_kernel(params, prev_out, W, dW, prev_delta, curr_delta, db, layer_parallelize);
}

}  // namespace kernels
}  // namespace core
}  // namespace tiny_dnn
