/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/backend.h"
#include "tiny_dnn/core/params/maxpool_params.h"

namespace tiny_dnn {
namespace kernels {

template <typename S1, typename S2>
inline void maxpool_op_nnpack(const Tensor<float, S1> &in_data,
                              Tensor<float, S2> &out_data,
                              const core::maxpool_params &params) {
#ifdef CNN_USE_NNPACK
  // call singleton to initialize NNPACK
  NNPackInitializer::getInstance().initialize();

  // TODO(edgarriba): recheck after paramters refactor
  const size_t input_channels = params.in.depth_;

  // TODO(edgarriba): recheck after paramters refactor
  const nnp_size input_size = {params.in.width_, params.in.height_};

  // input is already padded, so no need to do padding.
  // we'll assume that padding is symmetric
  const nnp_padding padding = {0.0, 0.0, 0.0, 0.0};

  // TODO(edgarriba): we should get this from maxpool_layer_params
  const nnp_size pooling_size = {params.pool_size_x, params.pool_size_y};

  // TODO(edgarriba): we should get this from maxpool_layer_params
  const nnp_size pooling_stride = {params.stride_x, params.stride_y};

  const float *input_ptr = in_data.host_pbegin();
  float *output_ptr      = out_data.host_pbegin();

  // initialize NNPACK threadpool with maximum number of threads
  NNPackThreadPool nnp_threadpool;
  nnp_threadpool.set_max_num_threads();
  nnp_threadpool.create();

  // TODO(edgarriba): we should get this from tensor shape. Ideally N from NCHW.
  const size_t batch_size = 1;

  const auto status = nnp_max_pooling_output(
    batch_size, input_channels, input_size, padding, pooling_size,
    pooling_stride, input_ptr, output_ptr, nnp_threadpool.threadpool());

  if (status != nnp_status_success) {
    throw nn_error("Could not succeed with nnp_max_pooling_output");
  }

  nnp_threadpool.destroy();

#else
  CNN_UNREFERENCED_PARAMETER(in_data);
  CNN_UNREFERENCED_PARAMETER(out_data);
  CNN_UNREFERENCED_PARAMETER(params);
  throw nn_error("tiny-dnn has not been compiled with NNPACK support.");
#endif
}

template <typename S1, typename S2>
inline void maxpool_op_nnpack(const Tensor<double, S1> &in_data,
                              Tensor<double, S2> &out_data,
                              const core::maxpool_params &params) {
  CNN_UNREFERENCED_PARAMETER(in_data);
  CNN_UNREFERENCED_PARAMETER(out_data);
  CNN_UNREFERENCED_PARAMETER(params);
  throw nn_error("Double isn't supported yet.");
}

}  // namespace kernels
}  // namespace tiny_dnn
