/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/params/fully_params.h"

#ifdef CNN_USE_CBLAS
extern "C" {
  #include <cblas.h>
}
#endif

namespace tiny_dnn {
namespace kernels {

inline void fully_connected_op_cblas(const tensor_t &in_data,
                                        const vec_t &W,
                                        const vec_t &bias,
                                        tensor_t &out_data,
                                        const core::fully_params &params,
                                        const bool layer_parallelize) {
#ifdef CNN_USE_CBLAS
  size_t M = 1;
  size_t N = params.out_size_;
  size_t K = params.in_size_;
  float_t alpha = 1;
  float_t beta = 1;
  float_t *input = (float_t *)in_data[0].data(); // [M*K]
  float_t *weight = (float_t *)W.data(); // [K*N]
  float_t *output = out_data[0].data();
  memcpy(output, bias.data(), sizeof(float_t) * N);
#ifdef CNN_USE_DOUBLE
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, input, K, weight, N, beta, output, N);
#else
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, input, K, weight, N, beta, output, N);
#endif

#endif // CNN_USE_CBLAS
}

}  // namespace kernels
}  // namespace tiny_dnn
