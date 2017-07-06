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

template <typename S1, typename S2, typename S3, typename S4>
inline void conv2d_op_nnpack(const Tensor<double, S1> &in_data,
                             const Tensor<double, S2> &weights,
                             const Tensor<double, S3> &bias,
                             Tensor<double, S4> &out_data,
                             const core::conv_params &params,
                             const bool layer_parallelize) {
  // fallback to tiny-backend when float_t is double
  conv2d_op_internal(in_data, weights, bias, out_data, params,
                     layer_parallelize);
}

template <typename S1, typename S2, typename S3, typename S4>
inline void conv2d_op_nnpack(const Tensor<float, S1> &in_data,
                             const Tensor<float, S2> &weights,
                             const Tensor<float, S3> &bias,
                             Tensor<float, S4> &out_data,
                             const core::conv_params &params,
                             const bool layer_parallelize) {
#ifdef CNN_USE_NNPACK
  // call singleton to initialize NNPACK
  core::NNPackInitializer::getInstance().initialize();

  // TODO(edgarriba): we should get this from conv_layer_params
  const auto algorithm                 = core::nnp_algorithm();
  const auto kernel_transform_strategy = core::nnp_kts();

  // TODO(edgarriba): that should be in/out shape
  const size_t input_channels  = params.in.depth_;
  const size_t output_channels = params.out.depth_;

  // input data passed by convolution layer has been padded
  // TODO(edgarriba): that should be in_data shape
  const nnp_size input_size = {params.in_padded.width_,
                               params.in_padded.height_};

  // TODO(edgarriba): that should be weights shape
  const nnp_size kernel_size = {params.weight.width_, params.weight.height_};

  // input is already padded, so no need to do padding.
  // we'll assume that padding is symmetric
  const nnp_padding padding = {0.0, 0.0, 0.0, 0.0};

  const nnp_size stride = {params.w_stride, params.h_stride};

  const float *input_ptr  = in_data.host_pointer(0, 0);
  const float *kernel_ptr = weights.host_pointer(0, 0);
  const float *bias_ptr   = bias.host_pointer(0, 0);

  float *output_ptr = out_data.host_pointer(0, 0);

  // initialize NNPACK threadpool with maximum number of threads
  core::NNPackThreadPool nnp_threadpool;
  nnp_threadpool.set_max_num_threads();
  nnp_threadpool.create();

  const auto status = nnp_convolution_inference(
    algorithm, kernel_transform_strategy, input_channels, output_channels,
    input_size, padding, kernel_size, stride, input_ptr, kernel_ptr, bias_ptr,
    output_ptr, nnp_threadpool.threadpool(), nullptr);

  if (status != nnp_status_success) {
    throw nn_error("Could not succeed with nnp_convolution_inference");
  }

  nnp_threadpool.destroy();

#else
  CNN_UNREFERENCED_PARAMETER(in_data);
  CNN_UNREFERENCED_PARAMETER(weights);
  CNN_UNREFERENCED_PARAMETER(bias);
  CNN_UNREFERENCED_PARAMETER(out_data);
  CNN_UNREFERENCED_PARAMETER(params);
  throw nn_error("tiny-dnn has not been compiled with NNPACK support.");
#endif
}

}  // namespace kernels
}  // namespace tiny_dnn
