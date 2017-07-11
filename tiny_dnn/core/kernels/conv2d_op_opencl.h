/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <algorithm>
#include <vector>

#include "tiny_dnn/core/framework/op_kernel.h"

namespace tiny_dnn {

class Conv2dOpenCLForwardOp : public core::OpKernel {
 public:
  explicit Conv2dOpenCLForwardOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
#if defined(USE_OPENCL) || defined(USE_CUDA)
    auto params = OpKernel::params_->conv();

    // incoming/outcoming data
    const tensor_t &in_data = context.input(0);
    const tensor_t &W       = context.input(1);
    const tensor_t &bias    = context.input(2);
    tensor_t &out_data      = context.output(1);

    // initialize outputs
    fill_tensor(out_data, float_t{0});

    // retrieve program from register
    CLCudaAPI::Program program = ProgramManager::getInstance().program(
      Program(context.device(), context.Layer()));
    nn_warn("Got Program");

    // Creates the kernel from the compiled program and sets the three
    // arguments.
    // Note that the indices of the arguments have to be set according to
    // their
    // order in the kernel.
    auto kernel = CLCudaAPI::Kernel(program, "CFMulti");
    nn_warn("Got Kernel");

    tiny_dnn::Device *device = context.device();
    CLCudaAPI::Context ctx   = context.device()->context();
    CLCudaAPI::Queue queue   = context.device()->queue();

    // TODO(edgar): check if we really need that
    for (size_t i = 0; i < in_data.size(); ++i) {
      // Creates device buffers and copies the host data to these
      // device buffers.

      auto dev_in = CLCudaAPI::Buffer<float_t>(ctx, queue, in_data[i].begin(),
                                               in_data[i].end());

      auto dev_W =
        CLCudaAPI::Buffer<float_t>(ctx, queue, W[0].begin(), W[0].end());

      auto dev_bias =
        CLCudaAPI::Buffer<float_t>(ctx, queue, bias[0].begin(), bias[0].end());

      auto dev_out = CLCudaAPI::Buffer<float_t>(ctx, queue, out_data[i].begin(),
                                                out_data[i].end());

      kernel.SetArgument(0, dev_in);    // image_data
      kernel.SetArgument(1, 0);         // image_offset
      kernel.SetArgument(2, dev_W);     // kernel_data
      kernel.SetArgument(3, 0);         // kernel_offset
      kernel.SetArgument(4, dev_bias);  // bias
      kernel.SetArgument(5, 0);         // bias_offset
      kernel.SetArgument(6, dev_out);   // convolved_image
      kernel.SetArgument(7, 0);         // convolved_image_offset

      kernel.SetArgument(8, static_cast<cl_ushort>(params.in.width_));  // WIDTH
      kernel.SetArgument(9,
                         static_cast<cl_ushort>(params.in.height_));  // HEIGHT
      kernel.SetArgument(
        10,
        static_cast<cl_ushort>(params.out.width_));  // OUTPUT_W
      kernel.SetArgument(
        11, static_cast<cl_ushort>(params.out.height_));  // OUTPUT_H

      // We make sure that work group size is multiple of 16
      size_t res  = device->device().MaxWorkGroupSize() % 16;
      size_t size = device->device().MaxWorkGroupSize() - res;

      auto global = std::vector<size_t>{size};
      auto local  = std::vector<size_t>{16};

      // Creates a new CLCudaAPI event to be able to time kernels
      auto event = CLCudaAPI::Event();

      // Enqueues the kernel and waits for the result.
      // Note that launching the kernel is always a-synchronous and thus
      // requires finishing the queue in order to complete the operation.
      nn_info("## Running the kernel ...");

      kernel.Launch(queue, global, local, event.pointer());
      queue.Finish(event);

      nn_info(" > Took " + to_string(event.GetElapsedTime()) + " ms");

      // Upload data GPU -> CPU
      std::vector<float_t> out(out_data[i].size(), 0);
      dev_out.Read(queue, out_data[i].size(), out);

      // FOR DEBUG ONLY
      nn_warn("output kernel");
      for (size_t j = 0; j < out.size(); ++j) {
        std::cout << out[j] << " ";
      }
      std::cout << std::endl;

      // copy back
      std::copy(std::begin(out), std::end(out),
                std::back_inserter(out_data[i]));
    }
#else
    CNN_UNREFERENCED_PARAMETER(context);
    throw nn_error("Not compiled with OpenCL");
#endif
  }
};

class Conv2dOpenCLBackwardOp : public core::OpKernel {
 public:
  explicit Conv2dOpenCLBackwardOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    CNN_UNREFERENCED_PARAMETER(context);
    nn_error("Not implemented yet.");
  }
};

}  // namespace tiny_dnn
