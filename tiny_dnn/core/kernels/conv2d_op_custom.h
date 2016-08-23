/*
    COPYRIGHT

    All contributions by Taiga Nomi
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.

    All other contributions:
    Copyright (c) 2013-2016, the respective contributors.
    All rights reserved.

    Each contributor holds copyright over their respective contributions.
    The project versioning (Git) records all such contribution source information.

    LICENSE

    The BSD 3-Clause License


    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name of tiny-dnn nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include "tiny_dnn/core/framework/op_kernel.h"

#include "tiny_dnn/core/kernels/conv2d.h"
#include "tiny_dnn/core/kernels/conv2d_op_custom_impl.h"

namespace tiny_dnn {

class Conv2dCustomForwardOp : private Conv2d, public core::OpKernel {
 public:
    explicit Conv2dCustomForwardOp(const core::OpKernelConstruction& context)
        : core::OpKernel(context) {}

    void compute(const core::OpKernelContext& context) override {
        // incomimg/outcoming data 
        const tensor_t& in_data = context.input(0);
        const vec_t&          W = context.input(1)[0];
        const vec_t&       bias = context.input(2)[0];
        tensor_t&      out_data = context.output(1);

        // pad input data
        tensor_t in_data_padded;
        Conv2d::setParams(OpKernel::params_);
        Conv2d::copy_and_pad_input(in_data, in_data_padded);

        // initialize outputs
        fill_tensor(out_data, float_t(0));

        // convolution algorithm
        conv2d_op_custom_impl(in_data_padded,
                              W,
                              bias,
                              out_data,
                              Conv2d::params(),
                              context.parallelize());
    }
};

class Conv2dCustomBackwardOp : private Conv2d, public core::OpKernel {
 public:
    explicit Conv2dCustomBackwardOp(const core::OpKernelConstruction& context)
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

        // initalize outputs
        fill_tensor(prev_delta, float_t(0));

        // convolution algorithm
        conv2d_op_custom_impl(prev_out, W, dW, db, curr_delta,
            prev_delta, Conv2d::params(), context.parallelize());

        // apply unpadding
        if (Conv2d::params().pad_type == core::padding::same) {
            Conv2d::copy_and_unpad_delta(prev_out, prev_delta);
        }
    }
};

}  // namespace tiny_dnn
