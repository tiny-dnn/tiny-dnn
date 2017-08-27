/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/framework/op_kernel.h"
#include "tiny_dnn/core/kernels/lstm_cell_op_internal.h"

namespace tiny_dnn {

class LSTMCellGradOp : public core::OpKernel {
 public:
  explicit LSTMCellGradOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    auto params = OpKernel::params_->lstm_cell();
    // incoming/outcoming data
    const tensor_t &x      = context.input(0);   // x
    const tensor_t &h_prev = context.input(1);   // h(t-1)
    const tensor_t &c_prev = context.input(2);   // c(t-1)
    const tensor_t &W_x2i  = context.input(3);   // W[x->i]
    const tensor_t &W_x2f  = context.input(4);   // W[x->f]
    const tensor_t &W_x2c  = context.input(5);   // W[x->c]
    const tensor_t &W_x2o  = context.input(6);   // W[x->o]
    const tensor_t &W_h2i  = context.input(7);   // W[h->i]
    const tensor_t &W_h2f  = context.input(8);   // W[h->f]
    const tensor_t &W_h2c  = context.input(9);   // W[h->c]
    const tensor_t &W_h2o  = context.input(10);  // W[h->o]

    tensor_t &d_o_prev = context.input_grad(0);   // dx
    tensor_t &d_h_prev = context.input_grad(1);   // dh_prev
    tensor_t &d_c_prev = context.input_grad(2);   // dc_prev
    tensor_t &dW_x2i   = context.input_grad(3);   // dW[x->i]
    tensor_t &dW_x2f   = context.input_grad(4);   // dW[x->f]
    tensor_t &dW_x2c   = context.input_grad(5);   // dW[x->c]
    tensor_t &dW_x2o   = context.input_grad(6);   // dW[x->o]
    tensor_t &dW_h2i   = context.input_grad(7);   // dW[h->i]
    tensor_t &dW_h2f   = context.input_grad(8);   // dW[h->f]
    tensor_t &dW_h2c   = context.input_grad(9);   // dW[h->c]
    tensor_t &dW_h2o   = context.input_grad(10);  // dW[h->o]
    tensor_t *db_2i    = params.has_bias_ ? &context.input_grad(11) : nullptr;
    tensor_t *db_2f    = params.has_bias_ ? &context.input_grad(12) : nullptr;
    tensor_t *db_2c    = params.has_bias_ ? &context.input_grad(13) : nullptr;
    tensor_t *db_2o    = params.has_bias_ ? &context.input_grad(14) : nullptr;

    const tensor_t &d_o_next = context.output_grad(0);  // d_o_next
    const tensor_t &d_h_next = context.output_grad(1);  // d_h_next
    const tensor_t &d_c_next = context.output_grad(2);  // d_c_next

    const tensor_t &o_next = context.output(0);
    const tensor_t &i      = context.output(3);
    const tensor_t &f      = context.output(4);
    const tensor_t &z      = context.output(5);
    const tensor_t &c      = context.output(6);  // pre_c is c_next
    tensor_t dummy;  // need lvalue for non-const reference

    // initialize outputs
    fill_tensor(d_o_prev, float_t{0});
    fill_tensor(d_h_prev, float_t{0});
    fill_tensor(d_c_prev, float_t{0});

    // call the algorithm depending on the selected engine type

    kernels::lstm_cell_op_internal(
      x, h_prev, c_prev, W_x2i[0], W_x2f[0], W_x2c[0], W_x2o[0], W_h2i[0],
      W_h2f[0], W_h2c[0], W_h2o[0], dW_x2i, dW_x2f, dW_x2c, dW_x2o, dW_h2i,
      dW_h2f, dW_h2c, dW_h2o, params.has_bias_ ? *db_2i : dummy,
      params.has_bias_ ? *db_2f : dummy, params.has_bias_ ? *db_2c : dummy,
      params.has_bias_ ? *db_2o : dummy, d_o_next, d_h_next, d_c_next, d_o_prev,
      d_h_prev, d_c_prev, o_next, i, f, z, c, params, context.parallelize());
  }
};

}  // namespace tiny_dnn
