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

#include "tiny_cnn/core/params/conv_params.h"

#ifdef USE_OPENCL
#include "third_party/CLCudaAPI/clpp11.h"
#endif

#ifdef CNN_USE_LIBDNN 
#include "libdnn.hpp"
#endif

namespace tiny_cnn {

class KernelLauncher {
 public:
    KernelLauncher() {}
};

class CLCudaAPIKernelLauncher : public KernelLauncher {
 public:
    CLCudaAPIKernelLauncher()
        : KernelLauncher()
        , kernel_(nullptr) {}

 private:
    std::unique_ptr<CLCudaAPI::Kernel> kernel_;
};

class LibDNNKernelLauncher : public KernelLauncher {
 public:
    LibDNNKernelLauncher()
        : KernelLauncher()
        , kernel_(nullptr) {}
 
    void tune(const int                device_id,
              const int                list_id,
              const core::conv_params& params,
              const cl_context         context,
              const cl_device_id       device,
              const cl_command_queue   queue) {
        // Context needs to be initialized with one device and queue
        greentea::device::setupViennaCLContext(device_id, context, device, queue);

        std::shared_ptr<greentea::device> dev_ptr =
            std::make_shared<greentea::device>(
                device_id, list_id, greentea::Backend::BACKEND_OpenCL);

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
        std::vector<int32_t> stride = { params.h_stride, params.w_stride };
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

        // Generate the libdnn kernels
        kernel_ = std::make_shared<greentea::LibDNNConv<float_t>>(config);
    }

 private:
    std::shared_ptr<greentea::LibDNNConv<float_t> > kernel_;
};

}  // namespace tiny_cnn
