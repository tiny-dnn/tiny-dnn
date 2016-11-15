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

#include "tiny_dnn/core/framework/op_kernel.h"

#ifdef CNN_USE_LIBDNN
#include "libdnn.hpp"
#endif

namespace tiny_dnn {

class Conv2dLibDNNForwardOp : public core::OpKernel {
 public:
    explicit Conv2dLibDNNForwardOp(const core::OpKernelConstruction& context)
            : core::OpKernel(context)
#ifdef CNN_USE_LIBDNN
            , initialized_(false)
#endif
    	{
        // TODO(edgar): remove this if statement when refactor
        // the init_backend() routine at layer level.
        if (OpKernel::device_ != nullptr) {
            auto params = OpKernel::params_->conv();
            init_libdnn(OpKernel::device_, params);
        }
    }

    void compute(const core::OpKernelContext& context) override {
#ifdef CNN_USE_LIBDNN
        // incoming/outcoming datm
        const tensor_t& in_data = context.input(0);
        const tensor_t&       W = context.input(1);
        const tensor_t&    bias = context.input(2);
        tensor_t&      out_data = context.output(1);

        // retrieve the convolutional parameters and pad input
        // Conv2d::setParams(context.params());

        // initialize outputs
        fill_tensor(out_data, float_t(0));

        // retrive device context and queue

        CLCudaAPI::Context ctx = OpKernel::device_->context();
        CLCudaAPI::Queue queue = OpKernel::device_->queue();

        for (serial_size_t i = 0; i < in_data.size(); ++i) {

            // allocate data to GPU

            auto dev_in = CLCudaAPI::Buffer<float_t>(ctx, queue,
                in_data[i].begin(), in_data[i].end());

            auto dev_W = CLCudaAPI::Buffer<float_t>(ctx, queue,
                W[0].begin(), W[0].end());

            auto dev_bias = CLCudaAPI::Buffer<float_t>(ctx, queue,
                bias[0].begin(), bias[0].end());

            auto dev_out = CLCudaAPI::Buffer<float_t>(ctx, queue,
                out_data[i].begin(), out_data[i].end());

            // cast data types and call libdnn

            // TODO(edgar): set a global variable with batch size or
            // embedd this inside the next gen Tensor class.
            const int batch_size = 1;

            const float_t* input_ptr   = double_cast(dev_in());
            const float_t* weights_ptr = double_cast(dev_W());
            const float_t* bias_ptr    = double_cast(dev_bias());

            float_t* output_ptr = mutable_double_cast(dev_out());

            // first time, tune the kernel

            // TODO(edgar/naibaf): enable when second generation
            // kernel are available

            if (!initialized_) {
                /*kernel_->Tune(const_cast<float_t*>(output_ptr), nullptr,
                              const_cast<float_t*>(weights_ptr), nullptr,
                              const_cast<float_t*>(bias_ptr), nullptr,
                              const_cast<float_t*>(input_ptr), nullptr,
                              batch_size);
                initialized_ = true;*/
            }

            // call libdnn forward

            kernel_->Forward(input_ptr,
                             weights_ptr,
                             bias_ptr,
                             output_ptr,
                             batch_size);


            // Upload data GPU -> CPU
            /*std::vector<float_t> dev_W_shadow(W.size(), 0);
            dev_W.Read(queue, W.size(), dev_W_shadow);

            // FOR DEBUG ONLY
            nn_warn("W kernel");
            for (serial_size_t j = 0; j < W.size(); ++j) {
                std::cout << dev_W_shadow[j] << " ";
            }
            std::cout << std::endl;

            // Upload data GPU -> CPU
            std::vector<float_t> dev_in_shadow(in_data_padded[i].size(), 0);
            dev_in.Read(queue, in_data_padded[i].size(), dev_in_shadow);

            // FOR DEBUG ONLY
            nn_warn("input kernel");
            for (serial_size_t j = 0; j < in_data_padded[i].size(); ++j) {
                std::cout << dev_in_shadow[j] << " ";
            }
            std::cout << std::endl;*/


            // Upload data GPU -> CPU
            // TODO(edgar): trigger this only when is needed
            std::vector<float_t> out(out_data[i].size(), 0);
            dev_out.Read(queue, out_data[i].size(), out);

            /*
            // FOR DEBUG ONLY
            nn_warn("output kernel");
            for (serial_size_t j = 0; j < out.size(); ++j) {
                std::cout << out[j] << " ";
            }
            std::cout << std::endl;
            */

            // copy data to be activated
            std::copy(std::begin(out), std::end(out), std::begin(out_data[i]));
        }

#else
        throw nn_error("TinyDNN was not compiled with LibDNN support.");
#endif
    }

 private:
#ifdef CNN_USE_LIBDNN
    float_t* mutable_double_cast(const cl_mem cl_mem_gpu) {
        return static_cast<float_t*>(
            reinterpret_cast<void*>(cl_mem_gpu));
    }

    const float_t* double_cast(const cl_mem cl_mem_gpu) {
        return reinterpret_cast<const float_t*>(
            reinterpret_cast<const void*>(cl_mem_gpu));
    }
#endif

    void init_libdnn(const Device* device, const core::conv_params& params) {
#ifdef CNN_USE_LIBDNN
        assert(device != nullptr);

        // Context needs to be initialized with one device and queue
        greentea::device::setupViennaCLContext(device->deviceId(),
            device->context()(), device->device()(), device->queue()());

        dev_ptr_ =
            std::make_shared<greentea::device>(
                device->deviceId(),
                device->deviceId(), /* list_id, */
                // TODO(edgar): refactor this since it's possible
                // to have OpenCL and CUDA.
#if defined(USE_OPENCL)
                greentea::Backend::BACKEND_OpenCL
#elif defined(USE_CUDA)
                greentea::Backend::BACKEND_CUDA
#else
                greentea::Backend::BACKEND_CPU
#endif
            );

        // Initialize device pointer in libdnn
        dev_ptr_->Init();

        // Setup libdnn params
        greentea::LibDNNConfig config;

        config.dev_ptr = dev_ptr_.get();

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
            dev_ptr_->CheckCapability("cl_khr_int64_base_atomics")) {
            config.wgalgo = greentea::LIBDNN_CONVOLUTION_WG_ALGO_ATOMIC;
            config.bwalgo = greentea::LIBDNN_CONVOLUTION_BW_ALGO_COL2IM_ATOMIC;
        } else {
            config.wgalgo = greentea::LIBDNN_CONVOLUTION_WG_ALGO_DIRECT;
            config.bwalgo = greentea::LIBDNN_CONVOLUTION_BW_ALGO_IM2COL;
        }

        // generate sources and compile kernel
        kernel_.reset(new greentea::LibDNNConv<float_t>(config));
#endif
    }

 private:
#ifdef CNN_USE_LIBDNN
    std::shared_ptr<greentea::device> dev_ptr_;
    std::shared_ptr<greentea::LibDNNConv<float_t> > kernel_;
    bool initialized_;
#endif
};

class Conv2dLibDNNBackwardOp : public core::OpKernel {
 public:
    explicit Conv2dLibDNNBackwardOp(const core::OpKernelConstruction& context)
        : core::OpKernel(context) {}

    void compute(const core::OpKernelContext& context) override {
        throw nn_error("Not implemented yet.");
    }
};

}  // namespace tiny_dnn
