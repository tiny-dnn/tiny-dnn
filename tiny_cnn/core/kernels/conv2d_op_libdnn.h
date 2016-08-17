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

    * Neither the name of tiny-cnn nor the names of its
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

#include "tiny_cnn/core/kernels/conv2d.h"
#include "tiny_cnn/core/framework/op_kernel.h"

#ifdef CNN_USE_LIBDNN
#include "libdnn.hpp"
#endif

namespace tiny_cnn {

class Conv2dLibDNNForwardOp : private Conv2d, public core::OpKernel {
 public:
    explicit Conv2dLibDNNForwardOp(const core::OpKernelConstruction& context)
            : core::OpKernel(context) {
        if (context.device() != nullptr) {
            Conv2d::setParams(context.params());
            init_libdnn(context.device(), Conv2d::params());
        }
    }

    void compute(const core::OpKernelContext& context) override {
#ifdef CNN_USE_LIBDNN
        // incoming/outcoming datm
        const tensor_t& in_data = context.input(0);
        const vec_t&          W = context.input(1)[0];
        const vec_t&       bias = context.input(2)[0];
        tensor_t&      out_data = context.output(1);

        // retrieve the convolutional parameters and pad input
        Conv2d::setParams(context.params());

        // initialize outputs
        fill_tensor(out_data, float_t(0));

        // pad input data
        tensor_t in_data_padded;
        Conv2d::copy_and_pad_input(in_data, in_data_padded);

#else
        throw nn_error("TinyDNN was not compiled with LibDNN support.");
#endif
    }

 private:
    void init_libdnn(const Device* device, const core::conv_params& params) {
        if (device == nullptr) {
            throw nn_error("no device ptr");
        } 

        nn_info("Device type: " + to_string(device->type()));
        nn_info("Device id: " + to_string(device->deviceId()));

        // Context needs to be initialized with one device and queue
        greentea::device::setupViennaCLContext(device->deviceId(),
            device->context()(), device->device()(), device->queue()());

        std::shared_ptr<greentea::device> dev_ptr =
            std::make_shared<greentea::device>(
                device->deviceId(),
                device->deviceId(), /* list_id, */
#if defined(USE_OPENCL)
                greentea::Backend::BACKEND_OpenCL
#elif defined(USE_CUDA)
                greentea::Backend::BACKEND_CUDA
#else
                greentea::Backend::BACKEND_CPU
#endif
            );

        // Initialize device pointer in libdnn
        dev_ptr->Init();

        // Setup libdnn params
        greentea::LibDNNConfig config;

        config.dev_ptr = dev_ptr.get();

        // NCHW shape setups

        const float_t dy = params.in_padded.height_ - params.in.height_;
        const float_t dx = params.in_padded.width_  - params.in.width_;

        std::vector<int32_t> in_shape = {
            1,
            params.in.depth_,
            params.in.height_,
            params.in.width_
        };

        std::vector<int32_t> out_shape = {
            1,
            params.out.depth_,
            params.out.height_,
            params.out.width_
        };

        std::vector<int32_t> kernel = {
            params.weight.height_,
            params.weight.width_
        };

        std::vector<int32_t> pad = { dy/2, dx/2 };

        std::vector<int32_t> stride = {
            params.h_stride,
            params.w_stride
        };

        std::vector<int32_t> dilation = { 1, 1 };

        config.in_shape  = in_shape;
        config.out_shape = out_shape;
        config.pad       = pad;
        config.kernel    = kernel;
        config.stride    = stride;
        config.dilation  = dilation;
        config.group     = 1;

        config.bias_term = params.has_bias;

        // Disables some optimizations but may give more stable results
        config.fast_unsafe_math = false;
        // Disables backward pass of weights during kernel.Backward();
        config.weights_backward = false;
        // Disables backward pass for bias during kernel.Backward();
        config.bias_backward    = false;

        // (Disabling bias and weight backward pass only propagates the data gradient (error))

        if (std::is_same<float_t, float>::value ||
            dev_ptr->CheckCapability("cl_khr_int64_base_atomics")) {
            config.wgalgo = greentea::LIBDNN_CONVOLUTION_WG_ALGO_ATOMIC;
            config.bwalgo = greentea::LIBDNN_CONVOLUTION_BW_ALGO_COL2IM_ATOMIC;
        } else {
            config.wgalgo = greentea::LIBDNN_CONVOLUTION_WG_ALGO_DIRECT;
            config.bwalgo = greentea::LIBDNN_CONVOLUTION_BW_ALGO_IM2COL;
        }

        // kernel_.reset(new greentea::LibDNNConv<float_t>(config));
        greentea::LibDNNConv<float_t> compute_kernel(config);

        std::cout << "End init_libdnn" << std::endl;
    }

 private:
#ifdef CNN_USE_LIBDNN
    std::shared_ptr<greentea::LibDNNConv<float_t> > kernel_;
#endif
};

class Conv2dLibDNNBackwardOp : private Conv2d, public core::OpKernel {
 public:
    explicit Conv2dLibDNNBackwardOp(const core::OpKernelConstruction& context)
        : core::OpKernel(context) {}

    void compute(const core::OpKernelContext& context) override {
        throw nn_error("Not implemented yet.");
    }
};

}  // namespace tiny_cnn
