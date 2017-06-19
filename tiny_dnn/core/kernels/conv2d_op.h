/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/framework/op_kernel.h"

#include "tiny_dnn/core/kernels/conv2d_op_avx.h"
#include "tiny_dnn/core/kernels/conv2d_op_internal.h"
#include "tiny_dnn/core/kernels/conv2d_op_nnpack.h"

namespace tiny_dnn {

class Conv2dOp : public core::OpKernel {
 public:
  explicit Conv2dOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    auto params = OpKernel::params_->conv();

    // TODO(Randl): Remove once layers forward and backward by themself.
    Tensor<float_t> in_data_t(context.input(0));
    const Tensor<float_t> weights_t(context.input(1)),
      bias_t =
        params.has_bias ? Tensor<float_t>(context.input(2)) : Tensor<float_t>();
    Tensor<float_t> out_data_t(context.output(0));

    // initialize outputs
    out_data_t.fill(0.0f);

    // call convolution algorithm depending
    // on the selected engine type

    const core::backend_t engine = context.engine();

    if (engine == core::backend_t::internal) {
      kernels::conv2d_op_internal(in_data_t, weights_t, bias_t, out_data_t,
                                  params, context.parallelize());

      // TODO(Randl): Remove once layers forward and backward by themself.
      context.output(0) = out_data_t.toTensor();
    } else if (engine == core::backend_t::nnpack) {
      kernels::conv2d_op_nnpack(in_data_t, weights_t, bias_t, out_data_t,
                                params, context.parallelize());

      // TODO(Randl): Remove once layers forward and backward by themself.
      context.output(0) = out_data_t.toTensor();
    } else if (engine == core::backend_t::avx) {
      kernels::conv2d_op_avx(in_data_t, weights_t, bias_t, out_data_t, params,
                             context.parallelize());

      // TODO(Randl): Remove once layers forward and backward by themself.
      context.output(0) = out_data_t.toTensor();
    } else {
      throw nn_error("Not supported engine: " + to_string(engine));
    }
  }
};

}  // namespace tiny_dnn
