// Copyright (c) 2013-2016, Taiga Nomi. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#pragma once

#include "tiny_dnn/core/framework/op_kernel.h"

#include "tiny_dnn/core/kernels/conv2d.h"
#include "tiny_dnn/core/kernels/conv2d_grad_op_avx.h"
#include "tiny_dnn/core/kernels/conv2d_op_custom.h"

namespace tiny_dnn {

class Conv2dGradOp : private Conv2d, public core::OpKernel {
 public:
    explicit Conv2dGradOp(const core::OpKernelConstruction& context)
        : core::OpKernel(context) {}

    void compute(const core::OpKernelContext& context) override {
        // incoming/outcoming data
        const tensor_t& prev_out = context.input(0);
        const vec_t& W  = context.input(1)[0];
        tensor_t&    dW = context.input_grad(1);
        tensor_t&    db = context.input_grad(2);
        tensor_t&    prev_delta = context.input_grad(0);
        tensor_t&    curr_delta = context.output_grad(1);

        // set an cast the convolutional parameters
        Conv2d::setParams(OpKernel::params_);

        // TODO(nyanp): Why we only need to initialize prev_delta ?

        // initalize outputs
        //fill_tensor(dW, float_t(0));
        //fill_tensor(db, float_t(0));
        fill_tensor(prev_delta, float_t(0));
        //fill_tensor(curr_delta, float_t(0));
        
        // call convolution algorithm depending
        // on the selected engine type

        const core::backend_t engine = context.engine();
        
        if (engine == core::backend_t::tiny_dnn) {
            kernels::conv2d_op_custom(
                prev_out,
                W,
                dW,
                db,
                curr_delta,
                prev_delta,
                Conv2d::params(),
                context.parallelize());
        }
        else if (engine == core::backend_t::avx) {
            kernels::conv2d_grad_op_avx(
                prev_out,
                W,
                dW,
                db,
                curr_delta,
                prev_delta,
                Conv2d::params(),
                context.parallelize());
        }
        else {
            throw nn_error("Not supported engine: " + to_string(engine));
        }
    }
};

}  // namespace tiny_dnn
