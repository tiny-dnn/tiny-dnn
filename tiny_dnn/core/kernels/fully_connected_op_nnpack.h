/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/backend.h"
#include "tiny_dnn/core/params/fully_params.h"

namespace tiny_dnn {
namespace kernels {
/**
 * Forward propogation for fully connected layer with NNPACK backend
 * @param in_data
 * @param weights
 * @param bias
 * @param out_data
 * @param params
 * @param layer_parallelize
 */
template <typename S1, typename S2, typename S3, typename S4>
inline void fully_connected_op_nnpack(const Tensor<float, S1> &in_data,
                                      const Tensor<float, S2> &weights,
                                      const Tensor<float, S3> &bias,
                                      Tensor<float, S4> &out_data,
                                      const bool layer_parallelize) {
#ifdef CNN_USE_NNPACK
  // call singleton to initialize NNPACK
  NNPackInitializer::getInstance().initialize();

  const float *kernel_ptr = weights.host_pointer(0, 0);
  const float *input_ptr  = in_data.host_pointer(0, 0);
  float *output_ptr       = out_data.host_pointer(0, 0);

  // TODO: embed it into a class
  const size_t num_mkl_threads = 1;
  pthreadpool_t threadpool     = pthreadpool_create(num_mkl_threads);

  size_t out_size = out_data.shape()[1], in_size = in_data.shape()[1];
  const auto status = nnp_fully_connected_inference(
    in_size, out_size, input_ptr, kernel_ptr, output_ptr, threadpool);

  if (status != nnp_status_success) {
    throw nn_error("Could not succeed with nnp_max_pooling_output");
  }

  // TODO: embed it into a class
  pthreadpool_destroy(threadpool);

  if (bias.size() > 0) {
    for_i(layer_parallelize, out_size,
          [&](size_t i) { output_ptr[i] += bias.host_at(0, i); });
  }
#else
  CNN_UNREFERENCED_PARAMETER(in_data);
  CNN_UNREFERENCED_PARAMETER(weights);
  CNN_UNREFERENCED_PARAMETER(bias);
  CNN_UNREFERENCED_PARAMETER(out_data);
  CNN_UNREFERENCED_PARAMETER(layer_parallelize);
  throw nn_error("TinyDNN has not been compiled with NNPACK support.");
#endif
}

/**
 * Forward propogation for fully connected layer with NNPACK backend
 * @param in_data
 * @param weights
 * @param bias
 * @param out_data
 * @param params
 * @param layer_parallelize
 */
template <typename S1, typename S2, typename S3, typename S4>
inline void fully_connected_op_nnpack(const Tensor<double, S1> &in_data,
                                      const Tensor<double, S2> &weights,
                                      const Tensor<double, S3> &bias,
                                      Tensor<double, S4> &out_data,
                                      const bool layer_parallelize) {
  // fallback to tiny-backend when float_t is double
  fully_connected_op_internal(in_data, weights, bias, out_data,
                              layer_parallelize);
}
}  // namespace kernels
}  // namespace tiny_dnn
