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

class LSTMCellOp : public core::OpKernel {
 public:
  explicit LSTMCellOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    auto params = OpKernel::params_->lstm_cell();

    // incomimg/outcoming data
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

    const tensor_t *b_2i = params.has_bias_ ? &context.input(11) : nullptr;
    const tensor_t *b_2f = params.has_bias_ ? &context.input(12) : nullptr;
    const tensor_t *b_2c = params.has_bias_ ? &context.input(13) : nullptr;
    const tensor_t *b_2o = params.has_bias_ ? &context.input(14) : nullptr;

    tensor_t &out_data = context.output(0);
    tensor_t &h_next   = context.output(1);
    tensor_t &c_next   = context.output(2);
    tensor_t &i        = context.output(3);
    tensor_t &f        = context.output(4);
    tensor_t &z        = context.output(5);
    tensor_t &c        = context.output(6);

    // initialize outputs
    fill_tensor(out_data, float_t{0});
    fill_tensor(h_next, float_t{0});
    fill_tensor(c_next, float_t{0});
    fill_tensor(i, float_t{0});
    fill_tensor(f, float_t{0});
    fill_tensor(z, float_t{0});
    fill_tensor(c, float_t{0});

    // call the algorithm depending  on the selected engine type

    const core::backend_t engine = context.engine();

    if (engine == core::backend_t::internal || engine == core::backend_t::avx) {
      kernels::lstm_cell_op_internal(
        x, h_prev, c_prev, W_x2i[0], W_x2f[0], W_x2c[0], W_x2o[0], W_h2i[0],
        W_h2f[0], W_h2c[0], W_h2o[0], params.has_bias_ ? (*b_2i)[0] : vec_t(),
        params.has_bias_ ? (*b_2f)[0] : vec_t(),
        params.has_bias_ ? (*b_2c)[0] : vec_t(),
        params.has_bias_ ? (*b_2o)[0] : vec_t(), out_data, h_next, c_next, i, f,
        z, c, params, context.parallelize());
    } else {
      throw nn_error("Not supported engine: " + to_string(engine));
    }
  }
};

}  // namespace tiny_dnn
