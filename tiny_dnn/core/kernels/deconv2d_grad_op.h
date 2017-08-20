/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/framework/op_kernel.h"
#include "tiny_dnn/core/kernels/deconv2d_op_internal.h"

namespace tiny_dnn {

class Conv2dTransposedGradOp : public core::OpKernel {
 public:
  explicit Conv2dTransposedGradOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    auto &params = OpKernel::params_->deconv();

    // TODO(Randl): Remove once layers forward and backward by themself.
    const Tensor<> prev_out(context.input(0));
    Tensor<> prev_delta(context.input_grad(0));
    Tensor<> curr_delta(context.output_grad(0));

    const Tensor<> weights(*(context.ith_parameter(0)->data()));
    const Tensor<> bias(params.has_bias ? *(context.ith_parameter(1)->data())
                                        : Tensor<>());

    Tensor<> weights_grads(*(context.ith_parameter(0)->grad()));
    Tensor<> bias_grads(params.has_bias ? *(context.ith_parameter(1)->grad())
                                        : Tensor<>());

    // initialize outputs
    prev_delta.fill(0);

    kernels::deconv2d_op_internal(prev_out, weights, weights_grads, bias_grads,
                                  curr_delta, prev_delta, params,
                                  params.has_bias, context.parallelize());
    context.ith_parameter(0)->set_grad(weights_grads);
    if (params.has_bias) {
      context.ith_parameter(1)->set_grad(bias_grads);
    }
    context.input_grad(0) = prev_delta;
  }
};

}  // namespace tiny_dnn
