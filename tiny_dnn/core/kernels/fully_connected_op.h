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
#include "tiny_dnn/core/kernels/fully_connected_op_nnpack.h"

namespace tiny_dnn {

class FullyConnectedOp : public core::OpKernel {
 public:
  explicit FullyConnectedOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    auto params = OpKernel::params_->fully();

    // incoming/outcoming data
    const xt::xarray<float_t> in_data = to_xtensor(context.input(0));
    const xt::xarray<float_t> W       = to_xtensor(context.input(1));
    const xt::xarray<float_t> B =
      params.has_bias_ ? to_xtensor(context.input(2)) : xt::xarray<float_t>();

    xt::xarray<float_t> out_data = to_xtensor(context.output(0));

    // initialize outputs
    out_data = xt::zeros<float_t>(out_data.shape());

    // call the algorithm depending  on the selected engine type

    const core::backend_t engine = context.engine();

    if (engine == core::backend_t::internal) {
      kernels::fully_connected_op_internal(
        in_data, xt::view(W, 0, xt::all()),
        params.has_bias_ ? xt::view(B, 0, xt::all()) : xt::xarray<float_t>(),
        out_data, params, context.parallelize());
    } else if (engine == core::backend_t::nnpack) {
      kernels::fully_connected_op_nnpack(
        in_data, xt::view(W, 0, xt::all()),
        params.has_bias_ ? xt::view(B, 0, xt::all()) : xt::xarray<float_t>(),
        out_data, params, context.parallelize());
    } else if (engine == core::backend_t::avx) {
      kernels::fully_connected_op_avx(
        in_data, xt::view(W, 0, xt::all()),
        params.has_bias_ ? xt::view(B, 0, xt::all()) : xt::xarray<float_t>(),
        out_data, params, context.parallelize());
    } else {
      throw nn_error("Not supported engine: " + to_string(engine));
    }
    context.output(0) = from_xtensor(out_data);
  }
};

}  // namespace tiny_dnn
