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

class GRUCellOp : public core::OpKernel {
 public:
  explicit GRUCellOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    auto params = OpKernel::params_->gru_cell();

    // incomimg/outcoming data
    const tensor_t &x      = context.input(0);  // x
    const tensor_t &h_prev = context.input(1);  // h(t-1)
    const tensor_t &W_x2z  = context.input(2);  // W[x->z]
    const tensor_t &W_x2r  = context.input(3);  // W[x->r]
    const tensor_t &W_x2h  = context.input(4);  // W[x->h]
    const tensor_t &W_hr2c = context.input(5);  // W[hr2c]
    const tensor_t &W_s2z  = context.input(6);  // W[s->z]
    const tensor_t &W_s2r  = context.input(7);  // W[s->r]

    const tensor_t *b_2z = params.has_bias_ ? &context.input(8) : nullptr;
    const tensor_t *b_2r = params.has_bias_ ? &context.input(9) : nullptr;
    const tensor_t *b_2h = params.has_bias_ ? &context.input(10) : nullptr;

    tensor_t &out    = context.output(0);  // output vector s(t)
    tensor_t &s      = context.output(1);  // s(t) is also next state
    tensor_t &h      = context.output(2);  // internal state  h(t)
    tensor_t &r      = context.output(3);  // reset gate    r(t)
    tensor_t &z      = context.output(4);  // update gate   z(t)
    tensor_t &hr     = context.output(5);  // aux state  pre_h(t)
    tensor_t &post_z = context.output(6);  // aux state  - post_z(t) (1-z)

    // initialize outputs
    fill_tensor(out, float_t{0});     // output vector s(t)
    fill_tensor(h, float_t{0});       // output state  h(t)
    fill_tensor(r, float_t{0});       // reset gate    r(t)
    fill_tensor(z, float_t{0});       // update gate   z(t)
    fill_tensor(hr, float_t{0});      // aux state  hr(t)
    fill_tensor(post_z, float_t{0});  // aux state  - post_z(t) (1-z)

    // call the algorithm depending  on the selected engine type

    const core::backend_t engine = context.engine();

    if (engine == core::backend_t::internal || engine == core::backend_t::avx) {
      kernels::gru_cell_op_internal(
        x, h_prev, W_x2z[0], W_x2r[0], W_x2h[0], W_hr2c[0], W_s2z[0], W_s2r[0],
        params.has_bias_ ? (*b_2z)[0] : vec_t(),
        params.has_bias_ ? (*b_2r)[0] : vec_t(),
        params.has_bias_ ? (*b_2h)[0] : vec_t(), out, h, r, z, hr, post_z,
        params, context.parallelize());
    } else {
      throw nn_error("Not supported engine: " + to_string(engine));
    }
    s = out;  // copy layer output to state
  }
};

}  // namespace tiny_dnn
