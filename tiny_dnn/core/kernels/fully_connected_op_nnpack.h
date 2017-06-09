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
                                      const fully_params &params,
                                      const bool layer_parallelize) {
#ifdef CNN_USE_NNPACK
  // call singleton to initialize NNPACK
  NNPackInitializer::getInstance().initialize();

  const float *kernel_ptr = &*weights.host_begin();
  const float *input_ptr  = in_data.host_begin();
  float *output_ptr       = out_data.host_begin();

  // TODO: embed it into a class
  const size_t num_mkl_threads = 1;
  pthreadpool_t threadpool     = pthreadpool_create(num_mkl_threads);

  const auto status =
    nnp_fully_connected_inference(params.in_size_, params.out_size_, input_ptr,
                                  kernel_ptr, output_ptr, threadpool);

  if (status != nnp_status_success) {
    throw nn_error("Could not succeed with nnp_max_pooling_output");
  }

  // TODO: embed it into a class
  pthreadpool_destroy(threadpool);

  if (params.has_bias_) {
    for_i(layer_parallelize, params.out_size_,
          [&](int i) { output_ptr[i] += bias[i]; });
  }
#else
  CNN_UNREFERENCED_PARAMETER(in_data);
  CNN_UNREFERENCED_PARAMETER(W);
  CNN_UNREFERENCED_PARAMETER(bias);
  CNN_UNREFERENCED_PARAMETER(out_data);
  CNN_UNREFERENCED_PARAMETER(params);
  CNN_UNREFERENCED_PARAMETER(layer_parallelize);
  throw nn_error("TinyDNN has not been compiled with NNPACK support.");
#endif
}

}  // namespace kernels
}  // namespace tiny_dnn
