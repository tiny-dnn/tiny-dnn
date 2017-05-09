/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/util/util.h"

#include "tiny_dnn/core/framework/op_kernel.h"

#include "tiny_dnn/core/kernels/fully_connected_op_avx.h"
#include "tiny_dnn/core/kernels/fully_connected_op_internal.h"

namespace tiny_dnn {

class FullyConnectedGradOp : public core::OpKernel {
 public:
  explicit FullyConnectedGradOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    auto params = OpKernel::params_->fully();

    // incoming/outcoming data
    const xt::xarray<float_t> prev_out = to_xtensor(context.input(0));
    const xt::xarray<float_t> W        = to_xtensor(context.input(1));
    xt::xarray<float_t> dW             = to_xtensor(context.input_grad(1));
    xt::xarray<float_t> dB             = to_xtensor(context.input_grad(2));
    xt::xarray<float_t> *db            = params.has_bias_ ? &dB : nullptr;
    xt::xarray<float_t> prev_delta     = to_xtensor(context.input_grad(0));
    xt::xarray<float_t> curr_delta     = to_xtensor(context.output_grad(0));
    xt::xarray<float_t> dummy;  // need lvalue for non-const reference

    // initialize outputs
    prev_delta = xt::zeros<float_t>(prev_delta.shape());

    // call the algorithm depending on the selected engine type

    const core::backend_t engine = context.engine();

    if (engine == core::backend_t::internal) {
      kernels::fully_connected_op_internal(
        prev_out, xt::view(W, 0, xt::all()), dW, params.has_bias_ ? *db : dummy,
        curr_delta, prev_delta, params, context.parallelize());
    } else if (engine == core::backend_t::avx) {
      kernels::fully_connected_op_avx(
        prev_out, xt::view(W, 0, xt::all()), dW, params.has_bias_ ? *db : dummy,
        curr_delta, prev_delta, params, context.parallelize());
    } else {
      throw nn_error("Not supported engine: " + to_string(engine));
    }
    context.input_grad(0) = from_xtensor(prev_delta);
    context.input_grad(1) = from_xtensor(dW);
      if(params.has_bias_)
        context.input_grad(2) = from_xtensor(dB); // TODO: temporary
  }
};

}  // namespace tiny_dnn
