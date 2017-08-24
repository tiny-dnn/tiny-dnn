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

class RecurrentCellGradOp : public core::OpKernel {
 public:
  explicit RecurrentCellGradOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    auto params = OpKernel::params_->recurrent_cell();

    Tensor<> dummy;
    // incoming/outcoming data
    const Tensor<> &prev_out(context.input(0));
    const Tensor<> &h(context.input(1));
    const Tensor<> &out_state(context.output(1));

    Tensor<> &prev_output_delta(context.input_grad(0));
    Tensor<> &prev_state_delta(context.input_grad(1));
    Tensor<> &curr_output_delta(context.output_grad(0));
    Tensor<> &curr_state_delta(context.output_grad(1));

    const Tensor<> &U(*(context.ith_parameter(0)->data()));
    const Tensor<> &W(*(context.ith_parameter(1)->data()));
    const Tensor<> &V(*(context.ith_parameter(2)->data()));

    Tensor<> &dU(*(context.ith_parameter(0)->grad()));
    Tensor<> &dW(*(context.ith_parameter(1)->grad()));
    Tensor<> &dV(*(context.ith_parameter(2)->grad()));

    Tensor<> &db =
      params.has_bias_ ? *(context.ith_parameter(3)->grad()) : dummy;
    Tensor<> &dc =
      params.has_bias_ ? *(context.ith_parameter(4)->grad()) : dummy;

    // initialize outputs
    prev_output_delta.fill(float_t(0.0));
    prev_state_delta.fill(float_t(0.0));
    // call the algorithm depending on the selected engine type
    kernels::recurrent_cell_op_internal(
      prev_out, h, U, W, V, dU, dW, dV, db, dc, curr_output_delta,
      curr_state_delta, prev_output_delta, prev_state_delta, out_state, params,
      context.parallelize());
  }
};

}  // namespace tiny_dnn
