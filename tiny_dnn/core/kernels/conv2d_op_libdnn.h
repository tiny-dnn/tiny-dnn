/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/framework/op_kernel.h"

#ifdef CNN_USE_LIBDNN
#include "libdnn.hpp"
#endif

namespace tiny_dnn {

class Conv2dLibDNNForwardOp : public core::OpKernel {
 public:
  explicit Conv2dLibDNNForwardOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context)
#ifdef CNN_USE_LIBDNN
      ,
      initialized_(false)
#endif
  {
    // TODO(edgar): remove this if statement when refactor
    // the init_backend() routine at layer level.
    if (OpKernel::device_ != nullptr) {
      auto params = OpKernel::params_->conv();
      init_libdnn(OpKernel::device_, params);
    }
  }

  void compute(core::OpKernelContext &context) override {
#ifdef CNN_USE_LIBDNN

    // retrieve the convolutional parameters and pad input
    // Conv2d::setParams(context.params());

    // TODO(Randl): Remove once layers forward and backward by themself.
    Tensor<float_t> in_data_t(context.input(0));
    const Tensor<float_t> weights_t(context.input(1)),
      bias_t = Tensor<float_t>(context.input(2));  // TODO has_bias
    Tensor<float_t> out_data_t(context.output(0));

    // initialize outputs
    out_data_t.fill(0.0f);

    // retrive device context and queue

    CLCudaAPI::Context ctx = OpKernel::device_->context();
    CLCudaAPI::Queue queue = OpKernel::device_->queue();

    size_t samples_num = in_data_t.shape()[0], out_size = out_data_t.shape()[1],
           in_size = in_data_t.shape()[1];
    for (size_t i = 0; i < samples_num; ++i) {
      auto in_begin  = in_data_t.host_pointer(i, 0),
           in_end    = in_data_t.host_pointer(i, 0) + in_size;
      auto out_begin = out_data_t.host_pointer(i, 0),
           out_end   = out_data_t.host_pointer(i, 0) + in_size;

      // allocate data to GPU
      auto dev_in = CLCudaAPI::Buffer<float_t>(ctx, queue, in_begin, in_end);

      auto dev_W = CLCudaAPI::Buffer<float_t>(
        ctx, queue, weights_t.host_pbegin(), weights_t.host_pend());

      auto dev_bias = CLCudaAPI::Buffer<float_t>(
        ctx, queue, bias_t.host_pbegin(), bias_t.host_pbegin());

      auto dev_out = CLCudaAPI::Buffer<float_t>(ctx, queue, out_begin, out_end);

      // cast data types and call libdnn

      // TODO(edgar): set a global variable with batch size or
      // embed this inside the next gen Tensor class.
      const int batch_size = 1;

      const float_t *input_ptr   = double_cast(dev_in());
      const float_t *weights_ptr = double_cast(dev_W());
      const float_t *bias_ptr    = double_cast(dev_bias());

      float_t *output_ptr = mutable_double_cast(dev_out());

      // first time, tune the kernel

      // TODO(edgar/naibaf): enable when second generation kernel are available

      if (!initialized_) {
        /*kernel_->Tune(const_cast<float_t*>(output_ptr), nullptr,
                      const_cast<float_t*>(weights_ptr), nullptr,
                      const_cast<float_t*>(bias_ptr), nullptr,
                      const_cast<float_t*>(input_ptr), nullptr,
                      batch_size);
        initialized_ = true;*/
      }

      // call libdnn forward

      kernel_->Forward(input_ptr, weights_ptr, bias_ptr, output_ptr,
                       batch_size);

      // Upload data GPU -> CPU
      /*std::vector<float_t> dev_W_shadow(W.size(), 0);
      dev_W.Read(queue, W.size(), dev_W_shadow);

      // FOR DEBUG ONLY
      nn_warn("W kernel");
      for (size_t j = 0; j < W.size(); ++j) {
          std::cout << dev_W_shadow[j] << " ";
      }
      std::cout << std::endl;

      // Upload data GPU -> CPU
      std::vector<float_t> dev_in_shadow(in_data_padded[i].size(), 0);
      dev_in.Read(queue, in_data_padded[i].size(), dev_in_shadow);

      // FOR DEBUG ONLY
      nn_warn("input kernel");
      for (size_t j = 0; j < in_data_padded[i].size(); ++j) {
          std::cout << dev_in_shadow[j] << " ";
      }
      std::cout << std::endl;*/

      // Upload data GPU -> CPU
      // TODO(edgar): trigger this only when is needed
      std::vector<float_t> out(out_size, 0);
      dev_out.Read(queue, out_size, out);

      /*
      // FOR DEBUG ONLY
      nn_warn("output kernel");
      for (size_t j = 0; j < out.size(); ++j) {
          std::cout << out[j] << " ";
      }
      std::cout << std::endl;
      */

      // copy data to be activated
      std::copy(std::begin(out), std::end(out), out_begin);
    }

#else
    CNN_UNREFERENCED_PARAMETER(context);
    throw nn_error("TinyDNN was not compiled with LibDNN support.");
#endif
  }

 private:
#ifdef CNN_USE_LIBDNN
  float_t *mutable_double_cast(const cl_mem cl_mem_gpu) {
    return static_cast<float_t *>(reinterpret_cast<void *>(cl_mem_gpu));
  }

  /**
   * Casts cl_mem to float_t pointer
   * @param cl_mem_gpu
   * @return
   */
  const float_t *double_cast(const cl_mem cl_mem_gpu) {
    return reinterpret_cast<const float_t *>(
      reinterpret_cast<const void *>(cl_mem_gpu));
  }
#endif

  void init_libdnn(const Device *device, const core::conv_params &params) {
#ifdef CNN_USE_LIBDNN
    assert(device != nullptr);

    // Context needs to be initialized with one device and queue
    greentea::device::setupViennaCLContext(
      device->deviceId(), device->context()(), device->device()(),
      device->queue()());

    dev_ptr_ = std::make_shared<greentea::device>(
      device->deviceId(), device->deviceId(), /* list_id, */
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
    const float_t dx = params.in_padded.width_ - params.in.width_;

    std::vector<int32_t> in_shape = {1, params.in.depth_, params.in.height_,
                                     params.in.width_};

    std::vector<int32_t> out_shape = {1, params.out.depth_, params.out.height_,
                                      params.out.width_};

    std::vector<int32_t> kernel = {params.weight.height_, params.weight.width_};

    std::vector<int32_t> pad = {dy / 2, dx / 2};

    std::vector<int32_t> stride = {params.h_stride, params.w_stride};

    std::vector<int32_t> dilation = {1, 1};

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
    config.bias_backward = false;

    // (Disabling bias and weight backward pass only propagates the data
    // gradient (error))

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
#else
    CNN_UNREFERENCED_PARAMETER(device);
    CNN_UNREFERENCED_PARAMETER(params);
#endif
  }

 private:
#ifdef CNN_USE_LIBDNN
  std::shared_ptr<greentea::device> dev_ptr_;
  std::shared_ptr<greentea::LibDNNConv<float_t>> kernel_;
  bool initialized_;
#endif
};

class Conv2dLibDNNBackwardOp : public core::OpKernel {
 public:
  explicit Conv2dLibDNNBackwardOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    CNN_UNREFERENCED_PARAMETER(context);
    throw nn_error("Not implemented yet.");
  }
};

}  // namespace tiny_dnn
