/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/framework/op_kernel.h"
#include "tiny_dnn/core/kernels/recurrent_cell_op_internal.h"

namespace tiny_dnn {

class RecurrentCellOp : public core::OpKernel {
 public:
  explicit RecurrentCellOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    auto params = OpKernel::params_->recurrent_cell();

    const Tensor<> dummy;
    // incoming/outcoming data
    const Tensor<> &in_data(context.input(0));
    const Tensor<> &prev_h(context.input(1));

    const Tensor<> &U(*(context.ith_parameter(0)->data()));
    const Tensor<> &W(*(context.ith_parameter(1)->data()));
    const Tensor<> &V(*(context.ith_parameter(2)->data()));

    const Tensor<> &b =
      params.has_bias_ ? *(context.ith_parameter(3)->data()) : dummy;
    const Tensor<> &c =
      params.has_bias_ ? *(context.ith_parameter(4)->data()) : dummy;
    Tensor<> &out_data(context.output(0));
    Tensor<> &next_h(context.output(1));

    // initialize outputs
    out_data.fill(float_t(0.0));
    next_h.fill(float_t(0.0));

    // call the algorithm depending  on the selected engine type
    const core::backend_t engine = context.engine();
    if (engine == core::backend_t::internal || engine == core::backend_t::avx) {
      kernels::recurrent_cell_op_internal(in_data, prev_h, U, W, V, b, c,
                                          out_data, next_h, params,
                                          context.parallelize());

    } else {
      throw nn_error("Not supported engine: " + to_string(engine));
    }
  }
};

}  // namespace tiny_dnn
