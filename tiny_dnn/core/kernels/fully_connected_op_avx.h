// Copyright (c) 2016, Taiga Nomi, Edgar Riba. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#pragma once

#include "tiny_dnn/core/kernels/fully_connected_op_custom.h"

namespace tiny_dnn {
namespace kernels {

inline void fully_connected_op_avx(const tensor_t& in_data, const vec_t& W,
                                   const vec_t& bias, tensor_t& out_data,
                                   const fully_params& params,
                                   const bool layer_parallelize) {
#ifdef CNN_USE_AVX
  // TODO(nyanp/beru): is this really AVX ??
  fully_connected_op_custom(in_data, W, bias, out_data, params,
                            layer_parallelize);
#else
  throw nn_error("TinyDNN has not been compiled with AVX support.");
#endif
}

inline void fully_connected_op_avx(const tensor_t& prev_out, const vec_t& W,
                                   tensor_t& dW, tensor_t& db,
                                   tensor_t& curr_delta, tensor_t& prev_delta,
                                   const fully_params& params,
                                   const bool layer_parallelize) {
#ifdef CNN_USE_AVX
  // TODO(nyanp/beru): is this really AVX ??
  fully_connected_op_custom(prev_out, W, dW, db, curr_delta, prev_delta, params,
                            layer_parallelize);
#else
  throw nn_error("TinyDNN has not been compiled with AVX support.");
#endif
}

}  // namespace kernels
}  // namespace tiny_dnn
