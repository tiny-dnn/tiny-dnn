/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/framework/op_kernel.h"

#include "tiny_dnn/core/kernels/fully_connected_op_avx.h"
#include "tiny_dnn/core/kernels/fully_connected_op_internal.h"
#include "tiny_dnn/core/kernels/fully_connected_op_nnpack.h"

namespace tiny_dnn {

class FullyConnectedOp : public core::OpKernel {
 public:
  explicit FullyConnectedOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    auto params = OpKernel::params_->fully();

    // incomimg/outcoming data
    // TODO(Randl): Remove once layers forward and backward by themself.
    const Tensor<float_t> in_data(context.input(0)), weights(context.input(1)),
      bias = params.has_bias_ ? Tensor<float_t>(context.input(2))
                              : Tensor<float_t>();
    Tensor<float_t> out_data(context.output(0));

    // initialize outputs
    out_data.fill(0);

    // call the algorithm depending  on the selected engine type

    const core::backend_t engine = context.engine();

    if (engine == core::backend_t::internal) {
      kernels::fully_connected_op_internal(in_data, weights, bias, out_data,
                                           context.parallelize());

      // TODO(Randl): Remove once layers forward and backward by themself.
      context.output(0) = out_data.toTensor();
    } else if (engine == core::backend_t::nnpack) {
      kernels::fully_connected_op_nnpack(in_data, weights, bias, out_data,
                                         context.parallelize());

      // TODO(Randl): Remove once layers forward and backward by themself.
      context.output(0) = out_data.toTensor();
    } else if (engine == core::backend_t::avx) {
      kernels::fully_connected_op_avx(in_data, weights, bias, out_data,
                                      context.parallelize());

      // TODO(Randl): Remove once layers forward and backward by themself.
      context.output(0) = out_data.toTensor();
    } else {
      throw nn_error("Not supported engine: " + to_string(engine));
    }
  }
};

}  // namespace tiny_dnn
