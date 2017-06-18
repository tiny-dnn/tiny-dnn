/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once
#include "tiny_dnn/core/backend.h"

#include "tiny_dnn/core/kernels/deconv2d_grad_kernel_avx.h"
#include "tiny_dnn/core/kernels/deconv2d_kernel_avx.h"

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

  // deconvolution
  //TODO: missing backward_activation
  avx_backend(deconv_params *params,
              std::function<void(const tensor_t &)> f1,
              std::function<void(const tensor_t &, tensor_t &)> f2,
              deconv_layer_worker_specific_storage *ptr)
    : params_d_(params),
      deconv_layer_worker_storage_(ptr),
      copy_and_unpad_output(f1),
      copy_and_pad_delta(f2) {}

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

  void deconv2d(const std::vector<tensor_t *> &in_data,
                std::vector<tensor_t *> &out_data) override {
    (*deconv_layer_worker_storage_).prev_out_ = in_data[0];
    tensor_t &out                             = *out_data[0];

    const Tensor<float_t> in_d(*in_data[0]), weights(((*in_data[1])[0])),
      bias = Tensor<float_t>((*in_data[2])[0]);
    Tensor<float_t> out_d(*out_data[0]);

    fill_tensor(
      out, float_t{0},
      params_d_->out.size());  // deconv2d-kernel requires padded size buffer
    out_d.fill(0);

    kernels::avx_deconv2d_kernel(in_d, weights, bias, out_d, *params_d_,
                                 layer_->parallelize());
    // TODO(Randl): Remove once layers forward and backward by themself.
    *out_data[0] = out_d.toTensor();

    copy_and_unpad_output(out);
    out = *(*deconv_layer_worker_storage_).curr_out_unpadded_;
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

  void deconv2d(const std::vector<tensor_t *> &in_data,
                const std::vector<tensor_t *> &out_data,
                std::vector<tensor_t *> &out_grad,
                std::vector<tensor_t *> &in_grad) override {
    CNN_UNREFERENCED_PARAMETER(out_data);
    deconv_layer_worker_specific_storage &cws = (*deconv_layer_worker_storage_);
    if (params_d_->pad_type == padding::same)
      copy_and_pad_delta(cws.curr_delta_padded, *in_grad[0]);

    tensor_t &curr_delta = (params_d_->pad_type == padding::same)
                             ? cws.curr_delta_padded
                             : *out_grad[0];

    backward_activation(*out_grad[0], *out_data[0], curr_delta);

    const Tensor<float_t> prev_out(*(cws.prev_out_)), weights((*in_data[1])[0]);
    Tensor<float_t> weights_grads(*in_grad[1]);
    Tensor<float_t> bias_grads = Tensor<float_t>(*in_grad[2]);
    Tensor<float_t> prev_delta(*in_grad[0]);
    Tensor<float_t> curr_d((params_d_->pad_type == padding::same)
                             ? cws.curr_delta_padded
                             : *out_grad[0]);

    assert(weights.size() == params_d_->weight.size());
    assert(weights_grads.shape()[1] == params_d_->weight.size());
    assert(curr_d.shape()[1] == layer_->out_shape()[0].size());

    // initialize outputs
    prev_delta.fill(float_t{0});

    kernels::tiny_deconv2d_back_kernel(prev_out, weights, weights_grads,
                                       bias_grads, curr_d, prev_delta,
                                       *params_d_);
    *in_grad[0] = prev_delta.toTensor();
    *in_grad[1] = weights_grads.toTensor();
    *in_grad[2] = bias_grads.toTensor();
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
