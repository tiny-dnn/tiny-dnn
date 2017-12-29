/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/framework/op_kernel.h"
#include "tiny_dnn/core/kernels/gru_cell_op_internal.h"

namespace tiny_dnn {

class GRUCellGradOp : public core::OpKernel {
 public:
  explicit GRUCellGradOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    auto params = OpKernel::params_->gru_cell();
    // incoming/outcoming data
    const tensor_t &x      = context.input(0);  // x
    const tensor_t &h_prev = context.input(1);  // h(t-1)
    const tensor_t &W_x2z  = context.input(2);  // W[x->z]
    const tensor_t &W_x2r  = context.input(3);  // W[x->r]
    const tensor_t &W_x2h  = context.input(4);  // W[x->h]
    const tensor_t &W_hr2c = context.input(5);  // W[hr2c]
    const tensor_t &W_s2z  = context.input(6);  // W[s->z]
    const tensor_t &W_s2r  = context.input(7);  // W[s->r]

    tensor_t &d_x_prev = context.input_grad(0);  // dx
    tensor_t &d_h_prev = context.input_grad(1);  // dh_prev
    tensor_t &dW_x2z   = context.input_grad(2);  // dW[x->z]
    tensor_t &dW_x2r   = context.input_grad(3);  // dW[x->r]
    tensor_t &dW_x2h   = context.input_grad(4);  // dW[x->h]
    tensor_t &dW_hr2c  = context.input_grad(5);  // dW[hr2c]
    tensor_t &dW_s2z   = context.input_grad(6);  // dW[s->z]
    tensor_t &dW_s2r   = context.input_grad(7);  // dW[s->r]
    tensor_t *db_2z    = params.has_bias_ ? &context.input_grad(8) : nullptr;
    tensor_t *db_2r    = params.has_bias_ ? &context.input_grad(9) : nullptr;
    tensor_t *db_2h    = params.has_bias_ ? &context.input_grad(10) : nullptr;

    const tensor_t &d_o_next = context.output_grad(0);  // d_o_next
    tensor_t &d_s_next       = context.output_grad(1);  // d_s_next

    const tensor_t &h      = context.output(2);  // internal state  h(t)
    const tensor_t &r      = context.output(3);  // reset gate      r(t)
    const tensor_t &z      = context.output(4);  // update gate     z(t)
    const tensor_t &hr     = context.output(5);  // aux state  h(t)*r(t)
    const tensor_t &post_z = context.output(6);  // aux state  - post_z(t) (1-z)
    tensor_t dummy;  // need lvalue for non-const reference

    // initialize input gradients
    fill_tensor(d_x_prev, float_t{0});
    fill_tensor(d_h_prev, float_t{0});

    // call the algorithm depending on the selected engine type

    kernels::gru_cell_op_internal(
      x, h_prev, W_x2z[0], W_x2r[0], W_x2h[0], W_hr2c[0], W_s2z[0], W_s2r[0],
      dW_x2z, dW_x2r, dW_x2h, dW_hr2c, dW_s2z, dW_s2r,
      params.has_bias_ ? *db_2z : dummy, params.has_bias_ ? *db_2r : dummy,
      params.has_bias_ ? *db_2h : dummy, d_o_next, d_s_next, d_x_prev, d_h_prev,
      h, r, z, hr, post_z, params, context.parallelize());
  }
};

}  // namespace tiny_dnn
