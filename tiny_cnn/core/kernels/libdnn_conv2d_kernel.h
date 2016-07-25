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

#include "tiny_cnn/core/params/conv_params.h"

#ifdef CNN_USE_LIBDNN
#include "libdnn.hpp"
//#include "viennacl/backend/opencl.hpp"
//#include "viennacl/ocl/backend.hpp"
//#include "viennacl/ocl/context.hpp"
//#include "viennacl/ocl/device.hpp"
//#include "viennacl/ocl/platform.hpp"

namespace tiny_cnn {
namespace core {
namespace kernels {

void libdnn_conv2d_kernel(const conv_params& params,
                          const vec_t&       in,
                          const vec_t&       W,
                          const vec_t&       bias,
                          vec_t&             a) {
    // instantiate pointer to device
    const int id = 0;
    const int list_id = 0;

    std::shared_ptr<greentea::device> dev_ptr =
        std::make_shared<greentea::device>(
            id, list_id, greentea::Backend::BACKEND_OpenCL);

    //viennacl::ocl::context &ctx = viennacl::ocl::get_context(id);

    /*cl_platform_id platform;
    cl_device_id device;
 
    // get first available platform
    clGetPlatformIDs(1, &platform, NULL);
 
    // get first available gpu device
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
 
    // create context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

    cl_command_queue queue;
    queue = clCreateCommandQueue(context, device,
        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, NULL);*/
   
    // error: ‘class greentea::device’ has no member named ‘setupViennaCLContext’
    // dev_ptr->setupViennaCLContext(id, context, device, queue);

    std::cout << "before Init()" << std::endl;
    dev_ptr->Init();
 
    // setup libdnn params
    greentea::LibDNNConfig config;

    config.dev_ptr = dev_ptr.get();

    config.in_shape[0] = params.in.depth_;
    config.in_shape[1] = params.in.height_;
    config.in_shape[2] = params.in.width_;
 
    config.out_shape[0] = params.out.depth_;
    config.out_shape[1] = params.out.height_;
    config.out_shape[2] = params.out.width_;

    config.kernel[0] = params.weight.width_;
    // config.kernel[1] = params.weight.height_;

    // const float_t dx = params.in_padded.width_  - params.in.width_;
    const float_t dy = params.in_padded.height_ - params.in.height_;

    config.pad[0] = static_cast<size_t>(dy/2);
    // config.pad[1] = static_cast<size_t>(dx/2);
    
    config.stride[0] = params.w_stride;
    // config.stride[1] = params.h_stride;

    config.bias_term = params.has_bias;

    config.fast_unsafe_math = false;
    config.weights_backward = false;
    config.bias_backward    = false;

    // call libdnn forward
    greentea::LibDNNConv<float_t> kernel(config);

    const float_t* input_ptr   = reinterpret_cast<const float_t*>(&in[0]);
    const float_t* weights_ptr = reinterpret_cast<const float_t*>(&W[0]);
    const float_t* bias_ptr    = reinterpret_cast<const float_t*>(&bias[0]);

    float_t* output_ptr = reinterpret_cast<float_t*>(&a[0]);
    
    const int batch_sz = 1;
    
    // call libdnn kernel
    kernel.Forward(input_ptr, weights_ptr, bias_ptr, output_ptr, batch_sz);
}

}  // namespace kernels
}  // namespace core
}  // namespace tiny_cnn

#endif
