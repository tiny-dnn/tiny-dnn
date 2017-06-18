/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/kernels/deconv2d_kernel_internal.h"
#include "tiny_dnn/core/params/deconv_params.h"

namespace tiny_dnn {
namespace core {
namespace kernels {

template <typename S1, typename S2, typename S3, typename S4>
inline void avx_deconv2d_kernel(const Tensor<float_t, S1> &in_data,
                                const Tensor<float_t, S2> &weights,
                                const Tensor<float_t, S3> &bias,
                                Tensor<float_t, S4> &out_data,
                                const deconv_params &params,
                                const bool layer_parallelize) {
  // fallback to non-avx version
  tiny_deconv2d_kernel(in_data, weights, bias, out_data, params,
                       layer_parallelize);
}

}  // namespace kernels
}  // namespace core
}  // namespace tiny_dnn
