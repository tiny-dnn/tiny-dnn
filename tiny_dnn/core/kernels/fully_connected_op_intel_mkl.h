/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/params/fully_params.h"

#ifdef CNN_USE_INTEL_MKL
extern "C" {
#include <mkl_cblas.h>
}
#endif

namespace tiny_dnn {
namespace kernels {

inline void fully_connected_op_intel_mkl(const tensor_t &in_data,
                                         const vec_t &W,
                                         const vec_t &bias,
                                         tensor_t &out_data,
                                         const core::fully_params &params,
                                         const bool layer_parallelize) {
#ifdef CNN_USE_INTEL_MKL
  if (in_data.size() != out_data.size()) {
    throw nn_error("fully_connected_op_intel_mkl args size mismatch");
  }
  for (size_t i = 0; i < in_data.size(); ++i) {
    size_t out_size       = params.out_size_;
    size_t in_size        = params.in_size_;
    float_t alpha         = 1;
    float_t beta          = 1;
    const float_t *input  = in_data[i].data();
    const float_t *weight = W.data();
    float_t *output       = out_data[i].data();
    if (bias.empty())
      memset(output, 0, sizeof(float_t) * out_size);
    else
      memcpy(output, bias.data(), sizeof(float_t) * out_size);
#ifdef CNN_USE_DOUBLE
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, out_size, in_size,
                alpha, input, in_size, weight, out_size, beta, output, out_size);
#else
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, out_size, in_size,
                alpha, input, in_size, weight, out_size, beta, output, out_size);
  }
#endif
#else
  throw nn_error("Compiled without Intel MKL support");
#endif  // CNN_USE_INTEL_MKL
}

}  // namespace kernels
}  // namespace tiny_dnn
