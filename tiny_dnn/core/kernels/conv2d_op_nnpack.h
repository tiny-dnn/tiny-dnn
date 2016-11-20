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

#include "tiny_dnn/core/params/conv_params.h"

#ifdef CNN_USE_NNPACK
#include "nnpack.h"

inline nnp_convolution_algorithm nnp_algorithm() {
    return nnp_convolution_algorithm_auto;
}

inline nnp_convolution_transform_strategy nnp_kts() {
    return nnp_convolution_transform_strategy_tuple_based;//some algorithm accept tuple based only
}
#endif

namespace tiny_dnn {
namespace kernels {

inline void
conv2d_op_nnpack(const tensor_t&         in_data,
                 const vec_t&                  W,
                 const vec_t&               bias,
                 tensor_t&              out_data,
                 const core::conv_params& params) {
#ifdef CNN_USE_NNPACK
    nnp_status init_status = nnp_initialize();
    if (init_status != nnp_status_success) {
        throw nn_error("Cannot initialize NNPACK.");
    }

    // TOOD: use input config
    const auto algorithm = nnp_algorithm();
    const auto kernel_transform_strategy = nnp_kts();

    const serial_size_t input_channels = params.in.depth_;
    const serial_size_t output_channels = params.out.depth_;

    //input data passed by convolution layer has been padded already
    //set input_size to padded size
    const nnp_size input_size = {
        static_cast<size_t>(params.in_padded.width_),
        static_cast<size_t>(params.in_padded.height_)
    };

    const nnp_size kernel_size = {
        static_cast<size_t>(params.weight.width_),
        static_cast<size_t>(params.weight.height_)
    };

    // input padded ,so no need to do padding
    const float_t dx =0;// params.in_padded.width_  - params.in.width_;
    const float_t dy =0;// params.in_padded.height_ - params.in.height_;

    // we'll assume that padding is symmetric

    const nnp_padding padding = {
        static_cast<size_t>(dy/2),  // top
        static_cast<size_t>(dx/2),  // right
        static_cast<size_t>(dy/2),  // bottom
        static_cast<size_t>(dx/2)   // left
    };

    const float* input_ptr  = reinterpret_cast<const float*>(in_data[0].data());
    const float* kernel_ptr = reinterpret_cast<const float*>(W.data());
    const float* bias_ptr   = reinterpret_cast<const float*>(bias.data());
    const nnp_size stride= {
        static_cast<size_t>(params.w_stride),
        static_cast<size_t>(params.h_stride)
    };

    float* output_ptr = out_data[0].data();

    // TODO: embed it into a class
    const size_t num_mkl_threads = 1;
    pthreadpool_t threadpool = pthreadpool_create(num_mkl_threads);

    nnp_profile* profile = nullptr;

    nnp_status status =
        nnp_convolution_inference(
            algorithm,
            kernel_transform_strategy,
            input_channels,
            output_channels,
            input_size,
            padding,
            kernel_size,
            stride,
            input_ptr,
            kernel_ptr,
            bias_ptr,
            output_ptr,
            threadpool,
            profile);

    if (status != nnp_status_success) {
        throw nn_error("Could not succeed with nnp_convolution_inference");
    }

    // TODO: embed it into a class
    pthreadpool_destroy(threadpool);
#else
    throw nn_error("TinyDNN has not been compiled with NNPACK support.");
#endif
}

}  // namespace kernels
}  // namespace tiny_dnn
