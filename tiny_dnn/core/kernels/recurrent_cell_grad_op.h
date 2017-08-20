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
    // incoming/outcoming data

    const Tensor<> prev_out(context.input(0)), h(context.input(1));
    const Tensor<> U(context.input(2)), W(context.input(3)),
      V(context.input(4));
    Tensor<> dU(context.input_grad(2)), dV(context.input_grad(3)),
      dW(context.input_grad(4));
    Tensor<> db =
               params.has_bias_ ? Tensor<>(context.input_grad(5)) : Tensor<>(),
             dc =
               params.has_bias_ ? Tensor<>(context.input_grad(6)) : Tensor<>();
    Tensor<> prev_output_delta(context.input_grad(0)),
      prev_state_delta(context.input_grad(1));
    Tensor<> curr_output_delta(context.output_grad(0)),
      curr_state_delta(context.output_grad(1));
    const Tensor<> out_state(context.output(1));

    // initialize outputs
    prev_output_delta.fill(float_t(0.0));
    prev_state_delta.fill(float_t(0.0));
    // call the algorithm depending on the selected engine type
    kernels::recurrent_cell_op_internal(
      prev_out, h, U, W, V, dU, dW, dV, db, dc, curr_output_delta,
      curr_state_delta, prev_output_delta, prev_state_delta, out_state, params,
      context.parallelize());

    // TODO(Randl): remove
    context.input_grad(2) = dU;
    context.input_grad(3) = dW;
    context.input_grad(4) = dV;
    if (params.has_bias_) {
      context.input_grad(5) = db;
      context.input_grad(6) = dc;
    }
    context.input_grad(0)  = prev_output_delta;
    context.input_grad(1)  = prev_state_delta;
    context.output_grad(0) = curr_output_delta;
    context.output_grad(1) = curr_state_delta;
  }
};

}  // namespace tiny_dnn
