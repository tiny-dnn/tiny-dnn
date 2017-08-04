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

    // incoming/outcoming data
    const Tensor<> in_data(context.input(0)), prev_h(context.input(1));
    const Tensor<> U(context.input(2)[0]), W(context.input(3)[0]),
      V(context.input(4)[0]);
    const Tensor<> bias = params.has_bias_ ? Tensor<>(context.input(5)[0])
                                           : Tensor<>(),
                   c = params.has_bias_ ? Tensor<>(context.input(6)[0])
                                        : Tensor<>();
    Tensor<> out_data(context.output(0)), next_h(context.output(1));

    // initialize outputs
    out_data.fill(float_t(0.0));
    next_h.fill(float_t(0.0));

    // call the algorithm depending  on the selected engine type
    const core::backend_t engine = context.engine();
    if (engine == core::backend_t::internal || engine == core::backend_t::avx) {
      kernels::recurrent_cell_op_internal(in_data, prev_h, U, W, V, bias, c,
                                          out_data, next_h, params,
                                          context.parallelize());

    } else {
      throw nn_error("Not supported engine: " + to_string(engine));
    }
    // TODO(Randl): remove
    context.output(0) = out_data;
    context.output(1) = next_h;
  }
};

}  // namespace tiny_dnn
