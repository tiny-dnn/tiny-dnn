/*
    Copyright (c) 2016, Taiga Nomi, Edgar Riba
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include "tiny_dnn/core/params/fully_params.h"

namespace tiny_dnn {
namespace kernels {

inline void
fully_connected_op_nnpack(const tensor_t&     in_data,
                          const vec_t&        W,
                          const vec_t&        bias,
                          tensor_t&           out_data,
                          const fully_params& params,
                          const bool          layer_parallelize) {
#ifdef CNN_USE_NNPACK
    const float* kernel_ptr = W.data();
    const float* input_ptr  = in_data[0].data();
    float*       output_ptr =out_data[0].data();

    // TODO: embed it into a class
    const size_t num_mkl_threads = 1;
    pthreadpool_t threadpool = pthreadpool_create(num_mkl_threads);

    const auto status =
        nnp_fully_connected_inference(
            params.in_size_,
            params.out_size_,
            input_ptr,
            kernel_ptr,
            output_ptr,
            threadpool);

     if (status != nnp_status_success) {
        throw nn_error("Could not succeed with nnp_max_pooling_output");
    }

    // TODO: embed it into a class
    pthreadpool_destroy(threadpool);

    // TODO: find a proper way to do this
    output_ptr =out_data[0].data();
    if (params.has_bias_) {
        for_i(layer_parallelize, params.out_size_, [&](int i) {
            // TODO(edgar): revise this
            // add bias manually (since no bias param in nnp_fully_connected_inference)
             output_ptr[i] += bias[i];
        });
    }
#else
    throw nn_error("TinyDNN has not been compiled with NNPACK support.");
#endif
}

}  // namespace kernels
}  // namespace tiny_dnn
