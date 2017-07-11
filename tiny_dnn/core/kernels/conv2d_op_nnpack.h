/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/backend.h"
#include "tiny_dnn/core/params/conv_params.h"

namespace tiny_dnn {
namespace kernels {

inline void conv2d_op_nnpack(const tensor_t &in_data,
                             const vec_t &W,
                             const vec_t &bias,
                             tensor_t &out_data,
                             const core::conv_params &params) {
#ifdef CNN_USE_NNPACK
  // call singleton to initialize NNPACK
  core::NNPackInitializer::getInstance().initialize();

  const auto algorithm                 = core::nnp_algorithm();
  const auto kernel_transform_strategy = core::nnp_kts();

  const size_t input_channels  = params.in.depth_;
  const size_t output_channels = params.out.depth_;

  // input data passed by convolution layer has been padded already
  // set input_size to padded size
  const nnp_size input_size = {params.in_padded.width_,
                               params.in_padded.height_};

  const nnp_size kernel_size = {params.weight.width_, params.weight.height_};

  // input padded ,so no need to do padding
  const float_t dx{0.0};  // params.in_padded.width_  - params.in.width_;
  const float_t dy{0.0};  // params.in_padded.height_ - params.in.height_;

  // we'll assume that padding is symmetric

  const nnp_padding padding = {
    dy / 2),  // top
    dx / 2),  // right
    dy / 2),  // bottom
    dx / 2)   // left
  };

  const nnp_size stride = {params.w_stride, params.h_stride};

  const float *input_ptr  = in_data[0].data();
  const float *kernel_ptr = W.data();
  const float *bias_ptr   = bias.data();

  float *output_ptr = out_data[0].data();

  // TODO(edgarriba): embed it into a class
  const size_t num_mkl_threads = 1;
  pthreadpool_t threadpool     = pthreadpool_create(num_mkl_threads);

  nnp_profile *profile = nullptr;

  nnp_status status = nnp_convolution_inference(
    algorithm, kernel_transform_strategy, input_channels, output_channels,
    input_size, padding, kernel_size, stride, input_ptr, kernel_ptr, bias_ptr,
    output_ptr, threadpool, profile);

  if (status != nnp_status_success) {
    throw nn_error("Could not succeed with nnp_convolution_inference");
  }

  // TODO(edgarriba): embed it into a class
  pthreadpool_destroy(threadpool);
#else
  CNN_UNREFERENCED_PARAMETER(in_data);
  CNN_UNREFERENCED_PARAMETER(W);
  CNN_UNREFERENCED_PARAMETER(bias);
  CNN_UNREFERENCED_PARAMETER(out_data);
  CNN_UNREFERENCED_PARAMETER(params);
  throw nn_error("TinyDNN has not been compiled with NNPACK support.");
#endif
}

}  // namespace kernels
}  // namespace tiny_dnn
