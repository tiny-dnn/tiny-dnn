/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/framework/op_kernel.h"

#include "tiny_dnn/core/kernels/maxpool_op_avx.h"
#include "tiny_dnn/core/kernels/maxpool_op_internal.h"

namespace tiny_dnn {

class MaxPoolGradOp : public core::OpKernel {
 public:
  explicit MaxPoolGradOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    auto &params = OpKernel::params_->maxpool();

    // incoming/outcoming data
    Tensor<float_t> prev_delta(context.input_grad(0));
    Tensor<float_t> curr_delta(context.output_grad(0));

    // initialize outputs
    prev_delta.fill(0.0f);

    // call the algorithm depending on the selected engine type

    const core::backend_t engine = context.engine();

    if (engine == core::backend_t::internal) {
      kernels::maxpool_grad_op_internal(prev_delta, curr_delta,
                                        params.out2inmax, params.in2out,
                                        context.parallelize());
      context.input_grad(0) = prev_delta.toTensor();
    } else if (engine == core::backend_t::avx) {
      kernels::maxpool_grad_op_avx(prev_delta, curr_delta, params.out2inmax,
                                   params.in2out, context.parallelize());
      context.input_grad(0) = prev_delta.toTensor();
    } else {
      throw nn_error("Not supported engine: " + to_string(engine));
    }
  }
};

}  // namespace tiny_dnn
