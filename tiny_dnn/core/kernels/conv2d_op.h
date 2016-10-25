// Copyright (c) 2013-2016, Taiga Nomi. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#pragma once

#include "tiny_dnn/core/framework/op_kernel.h"

#include "tiny_dnn/core/kernels/conv2d.h"
#include "tiny_dnn/core/kernels/conv2d_op_avx.h"
#include "tiny_dnn/core/kernels/conv2d_op_custom.h"
#include "tiny_dnn/core/kernels/conv2d_op_nnpack.h"

namespace tiny_dnn {

class Conv2dOp : private Conv2d, public core::OpKernel {
 public:
    explicit Conv2dOp(const core::OpKernelConstruction& context)
        : core::OpKernel(context) {}

    void compute(const core::OpKernelContext& context) override {
        // incomimg/outcoming data 
        const tensor_t& in_data = context.input(0);
        const vec_t&          W = context.input(1)[0];
        const vec_t&       bias = context.input(2)[0];
        tensor_t&      out_data = context.output(1);

        // initialize outputs
        fill_tensor(out_data, float_t(0));

        // set the convolution parameters
        Conv2d::setParams(OpKernel::params_);

        // call convolution algorithm depending
        // on the selected engine type

        const core::backend_t engine = context.engine();

        if (engine == core::backend_t::tiny_dnn) {
            kernels::conv2d_op_custom(
                in_data,
                W,
                bias,
                out_data,
                Conv2d::params(),
                context.parallelize());
        }
        else if (engine == core::backend_t::nnpack) {
            kernels::conv2d_op_nnpack(
                in_data,
                W,
                bias,
                out_data,
                Conv2d::params());
        }
        else if (engine == core::backend_t::avx) {
            kernels::conv2d_op_avx(
                in_data,
                W,
                bias,
                out_data,
                Conv2d::params(),
                context.parallelize());
        }
        else {
            throw nn_error("Not supported engine: " + to_string(engine));
        }
    }
};

}  // namespace tiny_dnn
