/*
    Copyright (c) 2017, Taiga Nomi
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/framework/op_kernel.h"
#include "tiny_dnn/core/kernels/global_avepool_op_internal.h"

namespace tiny_dnn {

class GlobalAvePoolOp : public core::OpKernel {
 public:
  explicit GlobalAvePoolOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    auto &params = OpKernel::params_->global_avepool();

    // incomimg / outcoming data
    const Tensor<float_t> in_data(context.input(0));
    Tensor<float_t> out_data(context.output(0));

    // initialize outputs
    out_data.fill(0.0f);

    // only internal kernel op implemented yet, so use it regardless
    // of the specified backend engine
    kernels::global_avepool_op_internal(in_data, out_data, params,
                                        context.parallelize());
    context.output(0) = out_data.toTensor();
  }
};

}  // namespace tiny_dnn
