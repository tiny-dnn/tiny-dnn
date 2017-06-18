/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/params/deconv_params.h"

namespace tiny_dnn {
namespace core {
namespace kernels {
template <typename S1,
          typename S2,
          typename S3,
          typename S4,
          typename S5,
          typename S6>
inline void avx_deconv2d_back_kernel(const Tensor<float_t, S1> &prev_out,
                                     const Tensor<float_t, S2> &weights,
                                     Tensor<float_t, S3> &weights_grads,
                                     Tensor<float_t, S4> &bias_grads,
                                     Tensor<float_t, S5> &curr_delta,
                                     Tensor<float_t, S6> &prev_delta,
                                     const deconv_params &params) {
  // fallback to non-avx version
  tiny_deconv2d_back_kernel(prev_out, weights, weights_grads, bias_grads,
                            curr_delta, prev_delta, params);
}

}  // namespace kernels
}  // namespace core
}  // namespace tiny_dnn
