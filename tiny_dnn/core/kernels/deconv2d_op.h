/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/framework/op_kernel.h"
#include "tiny_dnn/core/kernels/deconv2d_op_internal.h"

namespace tiny_dnn {

class Conv2dTransposedOp : public core::OpKernel {
 public:
  explicit Conv2dTransposedOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    auto &params = OpKernel::params_->deconv();

    // incomimg / outcoming data
    const Tensor<float_t> in_data(context.input(0));
<<<<<<< 13a5a8180703c5da82bbab42cea5dbf0fb69dcbf
    Tensor<float_t> out_data(context.output(0));

    const Tensor<float_t> weights(*(context.ith_parameter(0)->data()));
    const Tensor<float_t> bias(params.has_bias
                                 ? *(context.ith_parameter(1)->data())
                                 : Tensor<float_t>());

    // initialize outputs, pad first if unpadded already
    // output will be unpadded after forward pass calculation
    out_data.reshape({1, params.out.size()});
    out_data.fill(0);

    kernels::deconv2d_op_internal(in_data, weights, bias, out_data, params,
                                  params.has_bias, context.parallelize());
    // TODO(Randl): Remove once layers forward and backward by themself.
=======
    const Tensor<float_t> weights(context.input(1));
    const Tensor<float_t> bias =
      params.has_bias ? Tensor<float_t>(context.input(2)) : Tensor<float_t>();
    Tensor<float_t> out_data(context.output(0));

    // initialize outputs
    out_data.fill(0.0f);

    kernels::deconv2d_op_internal(in_data, weights, bias, out_data, params,
                                  params.has_bias, context.parallelize());
>>>>>>> Add OpKernel classes for deconv2d layer and remove old kernels. (#805)
    context.output(0) = out_data.toTensor();
  }
};

}  // namespace tiny_dnn
