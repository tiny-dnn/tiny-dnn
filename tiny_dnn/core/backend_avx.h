/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once
#include "tiny_dnn/core/backend.h"

namespace tiny_dnn {
namespace core {

class avx_backend : public backend {
 public:
  // context holds solution-dependent parameters
  // context should be able to hold any types of structures (like boost::any)

  // convolution
  avx_backend(conv_params *params,
              std::function<void(const tensor_t &)> f1,
              std::function<void(const tensor_t &, tensor_t &)> f2,
              conv_layer_worker_specific_storage *ptr)
    : /*params_c_(params),
      conv_layer_worker_storage_(ptr),*/
      copy_and_pad_input(f1),
      copy_and_unpad_delta(f2) {
    CNN_UNREFERENCED_PARAMETER(params);
    CNN_UNREFERENCED_PARAMETER(ptr);
  }

  // quantized convolution
  avx_backend(
    conv_params *params,
    std::function<void(const tensor_t &)> f1,
    std::function<void(const tensor_t &, tensor_t &)> f2,
    std::function<void(const tensor_t &, const tensor_t &, tensor_t &)> f3,
    conv_layer_worker_specific_storage *ptr)
    : /*params_c_(params),
      conv_layer_worker_storage_(ptr),*/
      copy_and_pad_input(f1),
      copy_and_unpad_delta(f2) {
    CNN_UNREFERENCED_PARAMETER(params);
    CNN_UNREFERENCED_PARAMETER(ptr);
  }

  // quantized deconvolution
  avx_backend(
    deconv_params *params,
    std::function<void(const tensor_t &)> f1,
    std::function<void(const tensor_t &, tensor_t &)> f2,
    std::function<void(const tensor_t &, const tensor_t &, tensor_t &)> f3,
    deconv_layer_worker_specific_storage *ptr)
    : params_d_(params),
      deconv_layer_worker_storage_(ptr),
      copy_and_unpad_output(f1),
      copy_and_pad_delta(f2),
      backward_activation(f3) {}

// maxpooling
#if 0
  avx_backend(std::vector<std::vector<size_t>> *out2in,
              std::vector<size_t> *in2out,
              max_pooling_layer_worker_specific_storage *ptr)
    : max_pooling_layer_worker_storage_(ptr),
      out2in_(out2in),
      in2out_(in2out) {}
#endif

  // fully_connected
  avx_backend(fully_params *params) /*: params_f_(params)*/ {
    CNN_UNREFERENCED_PARAMETER(params);
  }

  // quantized fully_connected
  avx_backend(
    fully_params *params,
    std::function<void(const tensor_t &, const tensor_t &, tensor_t &)> f)
    : /*params_f_(params),*/ backward_activation(f) {
    CNN_UNREFERENCED_PARAMETER(params);
  }

  // core math functions

  void conv2d_q(const std::vector<tensor_t *> &in_data,
                std::vector<tensor_t *> &out_data) override {
    CNN_UNREFERENCED_PARAMETER(in_data);
    CNN_UNREFERENCED_PARAMETER(out_data);
    throw nn_error("conv2d_q not implemented yet.");
  }

  void conv2d_eq(const std::vector<tensor_t *> &in_data,
                 std::vector<tensor_t *> &out_data) override {
    CNN_UNREFERENCED_PARAMETER(in_data);
    CNN_UNREFERENCED_PARAMETER(out_data);
    throw nn_error("conv2d_eq not implemented yet.");
  }

  void conv2d_q(const std::vector<tensor_t *> &in_data,
                const std::vector<tensor_t *> &out_data,
                std::vector<tensor_t *> &out_grad,
                std::vector<tensor_t *> &in_grad) override {
    CNN_UNREFERENCED_PARAMETER(in_data);
    CNN_UNREFERENCED_PARAMETER(out_data);
    CNN_UNREFERENCED_PARAMETER(out_grad);
    CNN_UNREFERENCED_PARAMETER(in_grad);
    throw nn_error("conv2d_q not implemented yet.");
  }

  void deconv2d_q(const std::vector<tensor_t *> &in_data,
                  std::vector<tensor_t *> &out_data) override {
    CNN_UNREFERENCED_PARAMETER(in_data);
    CNN_UNREFERENCED_PARAMETER(out_data);
    throw nn_error("not implemented yet.");
  }

  void deconv2d_eq(const std::vector<tensor_t *> &in_data,
                   std::vector<tensor_t *> &out_data) override {
    CNN_UNREFERENCED_PARAMETER(in_data);
    CNN_UNREFERENCED_PARAMETER(out_data);
    throw nn_error("not implemented yet.");
  }

  void deconv2d_q(const std::vector<tensor_t *> &in_data,
                  const std::vector<tensor_t *> &out_data,
                  std::vector<tensor_t *> &out_grad,
                  std::vector<tensor_t *> &in_grad) override {
    CNN_UNREFERENCED_PARAMETER(in_data);
    CNN_UNREFERENCED_PARAMETER(out_data);
    CNN_UNREFERENCED_PARAMETER(out_grad);
    CNN_UNREFERENCED_PARAMETER(in_grad);
    throw nn_error("not implemented yet.");
  }

  void fully_q(const std::vector<tensor_t *> &in_data,
               std::vector<tensor_t *> &out_data) override {
    CNN_UNREFERENCED_PARAMETER(in_data);
    CNN_UNREFERENCED_PARAMETER(out_data);
    throw nn_error("not implemented yet.");
  }

  void fully_eq(const std::vector<tensor_t *> &in_data,
                std::vector<tensor_t *> &out_data) override {
    CNN_UNREFERENCED_PARAMETER(in_data);
    CNN_UNREFERENCED_PARAMETER(out_data);
    throw nn_error("not implemented yet.");
  }

  void fully_q(const std::vector<tensor_t *> &in_data,
               const std::vector<tensor_t *> &out_data,
               std::vector<tensor_t *> &out_grad,
               std::vector<tensor_t *> &in_grad) override {
    CNN_UNREFERENCED_PARAMETER(in_data);
    CNN_UNREFERENCED_PARAMETER(out_data);
    CNN_UNREFERENCED_PARAMETER(out_grad);
    CNN_UNREFERENCED_PARAMETER(in_grad);
    throw nn_error("not implemented yet.");
  }

  backend_t type() const override { return backend_t::avx; }

 private:
  deconv_params *params_d_;
  deconv_layer_worker_specific_storage *deconv_layer_worker_storage_;
#if 0
  // Pointers to the convolution parameters
  conv_params *params_c_;
  fully_params *params_f_;

  // Pointers to the workers
  conv_layer_worker_specific_storage *conv_layer_worker_storage_;
  max_pooling_layer_worker_specific_storage *max_pooling_layer_worker_storage_;
  std::vector<std::vector<size_t>> *out2in_;
  std::vector<size_t> *in2out_;
#endif

  /* Pointers to parent class functions */
  std::function<void(const tensor_t &)> copy_and_pad_input;
  std::function<void(const tensor_t &)> copy_and_unpad_output;
  std::function<void(const tensor_t &, tensor_t &)> copy_and_unpad_delta;
  std::function<void(const tensor_t &, tensor_t &)> copy_and_pad_delta;
  std::function<void(const tensor_t &, const tensor_t &, tensor_t &)>
    backward_activation;
};

}  // namespace core
}  // namespace tiny_dnn
