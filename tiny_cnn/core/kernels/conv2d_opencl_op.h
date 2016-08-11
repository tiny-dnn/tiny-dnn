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

namespace tiny_cnn {

class Conv2dOpenCLForwardOp : private Conv2d, public core::OpKernel {
 public:
    explicit Conv2dOpenCLForwardOp(const core::OpKernelConstruction& context)
        : core::OpKernel(context) {}

    void compute(const core::OpKernelContext& context) override {
#if defined(USE_OPENCL) || defined(USE_CUDA)
        const tensor_t& in_data = context.input(0);
        const vec_t&          W = context.input(1)[0];
        const vec_t&       bias = context.input(2)[0];
        tensor_t&      out_data = context.output(1);

        Device* device = context.device();
        if (device == nullptr) {
            nn_warn("No device pointer");
        } else {
            nn_warn("Got device pointer");
        }

        CLCudaAPI::Program program = ProgramManager::getInstance()
            .program(Program(*context.device(), context.Layer()));
        nn_warn("Got Program");

        // TODO(edgar): it breaks here!!!

        // Creates the 'multiply' kernel from the compiled program and sets the three arguments. Note that
        // the indices of the arguments have to be set according to their order in the kernel.
        auto kernel = CLCudaAPI::Kernel(program, "CFMulti");
        nn_warn("Got Kernel");

        // retrieve the convolutional parameters and pad input
        Conv2d::setParams(context.params());

        // pad input data
        tensor_t in_data_padded;
        Conv2d::copy_and_pad_input(in_data, in_data_padded);

        // Creates two new device buffers and copies the host data to these device buffers.
        auto dev_in= CLCudaAPI::Buffer<float_t>(
                device->context(), device->queue(), in_data_padded.begin(), in_data_padded.end());
        auto dev_W = CLCudaAPI::Buffer<float_t>(
                device->context(), device->queue(), W.begin(), W.end());
        auto dev_bias = CLCudaAPI::Buffer<float_t>(
                device->context(), device->queue(), bias.begin(), bias.end());
        auto dev_out = CLCudaAPI::Buffer<float_t>(
                device->context(), device->queue(), out_data.begin(), out_data.end());

        kernel.SetArgument(0, dev_in);
        kernel.SetArgument(1, 0);
        kernel.SetArgument(2, dev_W);
        kernel.SetArgument(3, 0);
        kernel.SetArgument(4, dev_bias);
        kernel.SetArgument(5, 0);
        kernel.SetArgument(6, dev_out);
        kernel.SetArgument(7, 0);

        // TODO(edgar): how do we compute this value?
        cnn_size_t size = 100000;

        // Creates a 1-dimensional thread configuration with thread-blocks/work-groups of 256 threads
        // and a total number of threads equal to the number of elements in the input/output vectors.
        constexpr auto kWorkGroupSize = size_t{256};
        auto global = std::vector<size_t>{size};
        auto local = std::vector<size_t>{kWorkGroupSize};

        // Creates a new CLCudaAPI event to be able to time kernels
        auto event = CLCudaAPI::Event();

        // Enqueues the kernel and waits for the result. Note that launching the kernel is always
        // a-synchronous and thus requires finishing the queue in order to complete the operation.
        printf("## Running the kernel...\n");
        kernel.Launch(device->queue(), global, local, event.pointer());
        device->queue().Finish(event);
        printf(" > Took %.3lf ms\n", event.GetElapsedTime());
#else
        nn_error("Not compiled with OpenCL");
#endif
    }
};

class Conv2dOpenCLBackwardOp : private Conv2d, public core::OpKernel {
 public:
    explicit Conv2dOpenCLBackwardOp(const core::OpKernelConstruction& context)
        : core::OpKernel(context) {}

    void compute(const core::OpKernelContext& context) override {
        nn_error("Not implemented yet.");
    }
};

}  // namespace tiny_cnn
