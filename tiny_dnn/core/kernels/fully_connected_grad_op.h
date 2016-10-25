// Copyright (c) 2013-2016, Taiga Nomi. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#pragma once

#include "tiny_dnn/core/framework/op_kernel.h"

#include "tiny_dnn/core/kernels/fully_connected_op_avx.h"
#include "tiny_dnn/core/kernels/fully_connected_op_custom.h"

namespace tiny_dnn {

class FullyConnectedGradOp : public core::OpKernel {
 public:
    explicit FullyConnectedGradOp(const core::OpKernelConstruction& context)
        : core::OpKernel(context) {}

    void compute(const core::OpKernelContext& context) override {
        auto params = OpKernel::params_->fully();

        // incoming/outcoming data
        const tensor_t& prev_out = context.input(0);
        const tensor_t& W        = context.input(1);
        tensor_t& dW = context.input_grad(1);
        tensor_t* db = params.has_bias_ ? &context.input_grad(2) : nullptr;
        tensor_t& prev_delta = context.input_grad(0);
        tensor_t& curr_delta = context.output_grad(1);
        tensor_t dummy; // need lvalue for non-const reference

        // TODO(nyanp): Why we only need to initialize prev_delta ?

        // initialize outputs
        //fill_tensor(dW, float_t(0));
        //fill_tensor(db, float_t(0));
        fill_tensor(prev_delta, float_t(0));
        //fill_tensor(curr_delta, float_t(0));

        // call the algorithm depending on the selected engine type

        const core::backend_t engine = context.engine();

        if (engine == core::backend_t::tiny_dnn) {
            kernels::fully_connected_op_custom(
                prev_out,
                W[0],
                dW,
                params.has_bias_ ? *db : dummy,
                curr_delta,
                prev_delta,
                params,
                context.parallelize());
        }
        else if (engine == core::backend_t::avx) {
            kernels::fully_connected_op_avx(
                prev_out,
                W[0],
                dW,
                params.has_bias_ ? *db : dummy,
                curr_delta,
                prev_delta,
                params,
                context.parallelize());
        }
        else {
            throw nn_error("Not supported engine: " + to_string(engine));
        }
    }
};

}  // namespace tiny_dnn
