/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/framework/op_kernel.h"

#include "tiny_dnn/core/kernels/conv2d_grad_op_avx.h"
#include "tiny_dnn/core/kernels/conv2d_op_internal.h"

namespace tiny_dnn {

class Conv2dGradOp : public core::OpKernel {
 public:
  explicit Conv2dGradOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    auto params = OpKernel::params_->conv();

    // TODO(Randl): Remove once layers forward and backward by themself.
    const Tensor<float_t> prev_out_t(context.input(0)),
      weights_t(context.input(1));
    Tensor<float_t> weights_grads_t(context.input_grad(1));
    Tensor<float_t> bias_grads_t(context.input_grad(2));
    Tensor<float_t> prev_delta_t(context.input_grad(0));
    Tensor<float_t> curr_delta_t(context.output_grad(0));

    // initialize outputs
    prev_delta_t.fill(float_t{0});

    // call convolution algorithm depending
    // on the selected engine type

    const core::backend_t engine = context.engine();

    if (engine == core::backend_t::internal) {
      kernels::conv2d_op_internal(prev_out_t, weights_t, weights_grads_t,
                                  bias_grads_t, curr_delta_t, prev_delta_t,
                                  params, context.parallelize());
      context.input_grad(0) = prev_delta_t.toTensor();
      context.input_grad(1) = weights_grads_t.toTensor();
      context.input_grad(2) = bias_grads_t.toTensor();
    } else if (engine == core::backend_t::avx) {
      kernels::conv2d_op_internal(prev_out_t, weights_t, weights_grads_t,
                                  bias_grads_t, curr_delta_t, prev_delta_t,
                                  params, context.parallelize());
      context.input_grad(0) = prev_delta_t.toTensor();
      context.input_grad(1) = weights_grads_t.toTensor();
      context.input_grad(2) = bias_grads_t.toTensor();
    } else {
      throw nn_error("Not supported engine: " + to_string(engine));
    }
  }
};

}  // namespace tiny_dnn
