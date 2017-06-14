/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/framework/op_kernel.h"

#include "tiny_dnn/core/kernels/maxpool_op_avx.h"
#include "tiny_dnn/core/kernels/maxpool_op_internal.h"
#include "tiny_dnn/core/kernels/maxpool_op_nnpack.h"

namespace tiny_dnn {

class MaxPoolOp : public core::OpKernel {
 public:
  explicit MaxPoolOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    auto &params = OpKernel::params_->maxpool();

    const Tensor<float_t> in_data(context.input(0));
    Tensor<float_t> out_data(context.output(0));

    // initialize outputs
    out_data.fill(0.0f);

    // call convolution algorithm depending
    // on the selected engine type

    const core::backend_t engine = context.engine();

    if (engine == core::backend_t::internal) {
      kernels::maxpool_op_internal(in_data, out_data, params.out2inmax,
                                   params.out2in, context.parallelize());
      context.output(0) = out_data.toTensor();
    } else if (engine == core::backend_t::nnpack) {
      // NNPACK supports stride != 2 or pool_size !=2
      // there's optimization over stride=2 and pool_size=2
      /*
      if (params.stride_x != 2 || params.stride_y != 2) {
           throw nn_error("NNPACK Max-Pool requires a stride == 2.");
      }

      if (params.pool_size_x != 2 || params.pool_size_y != 2) {
           throw nn_error("NNPACK Max-Pool requires a pool size == 2.");
      }

      */
      kernels::maxpool_op_nnpack(in_data, out_data, params);
      context.output(0) = out_data.toTensor();
    } else if (engine == core::backend_t::avx) {
      kernels::maxpool_op_avx(in_data, out_data, params.out2inmax,
                              params.out2in, context.parallelize());
      // TODO(Randl): Remove once layers forward and backward by themself.
      context.output(0) = out_data.toTensor();
    } else {
      throw nn_error("Not supported engine: " + to_string(engine));
    }
  }
};

}  // namespace tiny_dnn
