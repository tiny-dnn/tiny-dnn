/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/framework/op_kernel.h"

#include "tiny_dnn/core/kernels/rnn_cell_op_internal.h"

namespace tiny_dnn {

class RecurrentCellGradOp : public core::OpKernel {
 public:
  explicit RecurrentCellGradOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    auto params = OpKernel::params_->rnn_cell();
    // incoming/outcoming data
    const tensor_t &prev_out = context.input(0);
    const tensor_t &h        = context.input(1);
    const tensor_t &U        = context.input(2);
    const tensor_t &W        = context.input(3);
    const tensor_t &V        = context.input(4);
    tensor_t &dU             = context.input_grad(2);
    tensor_t &dW             = context.input_grad(3);
    tensor_t &dV             = context.input_grad(4);
    tensor_t *db = params.has_bias_ ? &context.input_grad(5) : nullptr;
    tensor_t *dc = params.has_bias_ ? &context.input_grad(6) : nullptr;
    tensor_t &prev_output_delta = context.input_grad(0);
    tensor_t &prev_state_delta  = context.input_grad(1);
    tensor_t &curr_output_delta = context.output_grad(0);
    tensor_t &curr_state_delta  = context.output_grad(1);
    const tensor_t &out_state   = context.output(1);
    tensor_t dummy;  // need lvalue for non-const reference

    // initialize outputs
    fill_tensor(prev_output_delta, float_t{0});
    fill_tensor(prev_state_delta, float_t{0});

    // call the algorithm depending on the selected engine type

    kernels::rnn_cell_op_internal(
      prev_out, h, U[0], W[0], V[0], dU, dW, dV, params.has_bias_ ? *db : dummy,
      params.has_bias_ ? *dc : dummy, curr_output_delta, curr_state_delta,
      prev_output_delta, prev_state_delta, out_state, params,
      context.parallelize());
  }
};

}  // namespace tiny_dnn
