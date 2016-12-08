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

#include "tiny_dnn/core/kernels/maxpool_op_internal.h"
#include "tiny_dnn/core/kernels/maxpool_op_nnpack.h"
#include "tiny_dnn/core/kernels/maxpool_op_avx.h"

namespace tiny_dnn {

class MaxPoolOp : public core::OpKernel {
 public:
    explicit MaxPoolOp(const core::OpKernelConstruction& context)
        : core::OpKernel(context) {}

    void compute(const core::OpKernelContext& context) override {
        auto& params = OpKernel::params_->maxpool();

        // incomimg/outcoming data 
        const tensor_t& in_data = context.input(0);
        tensor_t&      out_data = context.output(1);

        // initialize outputs
        fill_tensor(out_data, float_t(0));

        // call convolution algorithm depending
        // on the selected engine type

        const core::backend_t engine = context.engine();

        if (engine == core::backend_t::internal) {
            kernels::maxpool_op_internal(
                in_data,
                out_data,
                params.out2inmax,
                params.out2in,
                context.parallelize());
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
	    kernels::maxpool_op_nnpack(
                in_data,
                out_data,
                params);
        } else if (engine == core::backend_t::avx) {
	    kernels::maxpool_op_avx(
                in_data,
                out_data,
                params.out2inmax,
                params.out2in,
                context.parallelize());
        } else {
            throw nn_error("Not supported engine: " + to_string(engine));
        }
    }
};

}  // namespace tiny_dnn
