/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <vector>

#include "tiny_dnn/config.h"
#include "tiny_dnn/core/backend.h"

#include "tiny_dnn/core/kernels/tiny_quantized_conv2d_kernel.h"
#include "tiny_dnn/core/kernels/tiny_quantized_deconv2d_kernel.h"

#ifdef CNN_USE_GEMMLOWP
#include "tiny_dnn/core/kernels/tiny_quantized_fully_connected_kernel.h"
#endif  // CNN_USE_GEMMLOWP

namespace tiny_dnn {
namespace core {

class tiny_backend : public backend {
 public:
  // context holds solution-dependent parameters
  // context should be able to hold any types of structures (like boost::any)

  // convolution
  tiny_backend(conv_params *params,
               std::function<void(const Tensor<> &)> f1,
               std::function<void(const Tensor<> &, Tensor<> &)> f2,
               conv_layer_worker_specific_storage *ptr)
    : params_c_(params),
      conv_layer_worker_storage_(ptr),
      copy_and_pad_input(f1),
      copy_and_unpad_delta(f2) {}

  // quantized convolution
  tiny_backend(
    conv_params *params,
    std::function<void(const Tensor<> &)> f1,
    std::function<void(const Tensor<> &, Tensor<> &)> f2,
    std::function<void(const Tensor<> &, const Tensor<> &, Tensor<> &)> f3,
    conv_layer_worker_specific_storage *ptr)
    : params_c_(params),
      conv_layer_worker_storage_(ptr),
      copy_and_pad_input(f1),
      copy_and_unpad_delta(f2) {}

  // deconvolution
  tiny_backend(deconv_params *params,
               std::function<void(const Tensor<> &)> f1,
               std::function<void(const Tensor<> &, Tensor<> &)> f2,
               deconv_layer_worker_specific_storage *ptr)
    : params_d_(params),
      deconv_layer_worker_storage_(ptr),
      copy_and_unpad_output(f1),
      copy_and_pad_delta(f2) {}

  // quantized deconvolution
  tiny_backend(
    deconv_params *params,
    std::function<void(const Tensor<> &)> f1,
    std::function<void(const Tensor<> &, Tensor<> &)> f2,
    std::function<void(const Tensor<> &, const Tensor<> &, Tensor<> &)> f3,
    deconv_layer_worker_specific_storage *ptr)
    : params_d_(params),
      deconv_layer_worker_storage_(ptr),
      copy_and_unpad_output(f1),
      copy_and_pad_delta(f2),
      backward_activation(f3) {}

  // fully_connected
  explicit tiny_backend(fully_params *params)
#if 0
    : params_f_(params)
#endif
  {
    CNN_UNREFERENCED_PARAMETER(params);
  }

  // quantized fully_connected
  tiny_backend(
    fully_params *params,
    std::function<void(const Tensor<> &, const Tensor<> &, Tensor<> &)> f)
    : /*params_f_(params),*/ backward_activation(f) {
    CNN_UNREFERENCED_PARAMETER(params);
  }

  // core math functions

  // quantized convolution
  void conv2d_q(const std::vector<Tensor<> *> &in_data,
                std::vector<Tensor<> *> &out_data) override {
    copy_and_pad_input(*in_data[0]);
    auto W        = layer_->parameter_at(0).data();
    auto bias     = params_c_->has_bias ? layer_->parameter_at(1).data() : Tensor<>();
    Tensor<> &out = *out_data[0];
    const std::vector<const Tensor<> *> &in =
      conv_layer_worker_storage_->prev_out_padded_;  // input // NOLINT

    out.fill(float_t(0.0));

    for (size_t i = 0; i < in.size(); i++) {
      kernels::tiny_quantized_conv2d_kernel(
        *params_c_, *in[i], W, bias,
        out.subView(TensorSingleIndex(i), TensorAll()), layer_->parallelize());
    }
  }

  /**
   * efficient quantization without abundant quantization/dequantization
   * @param in_data
   * @param out_data
   */
  void conv2d_eq(const std::vector<Tensor<> *> &in_data,
                 std::vector<Tensor<> *> &out_data) override {
    copy_and_pad_input(*in_data[0]);
    auto W    = layer_->parameter_at(0).data();
    auto bias = params_c_->has_bias ? layer_->parameter_at(1).data() : Tensor<>();
    const Tensor<> &in_r = *in_data[1];
    auto W_r        = layer_->parameter_at(params_c_->has_bias ? 2:1).data();
    auto b_r        =
        params_c_->has_bias ? layer_->parameter_at(3).data() : Tensor<>();
    Tensor<> &out   = *out_data[0];
    Tensor<> &out_r = *out_data[1];

    const std::vector<const Tensor<> *> &in =
      conv_layer_worker_storage_->prev_out_padded_;  // input // NOLINT

    out.fill(float_t(0.0));
    for (size_t i = 0; i < in.size(); i++) {
      kernels::tiny_quantized_conv2d_kernel(*params_c_, *in[i], W, bias,
                                            in_r[i], W_r, b_r, out[i], out_r[i],
                                            layer_->parallelize());
    }
  }

  void conv2d_q(const std::vector<Tensor<> *> &in_data,
                const std::vector<Tensor<> *> &out_data,
                std::vector<Tensor<> *> &out_grad,
                std::vector<Tensor<> *> &in_grad) override {
    conv_layer_worker_specific_storage &cws = (*conv_layer_worker_storage_);

    std::vector<const vec_t *> &prev_out = cws.prev_out_padded_;

    auto W = layer_->parameter_at(0).data();
    Tensor<> &dW         = layer_->parameter_at(0).grad();
    Tensor<> &db         = layer_->parameter_at(1).grad();
    Tensor<> &curr_delta = *out_grad[0];
    Tensor<> *prev_delta = (params_c_->pad_type == padding::same)
                             ? &cws.prev_delta_padded_
                             : in_grad[0];

    assert(W.size() == params_c_->weight.size());
    assert(dW[0].size() == params_c_->weight.size());
    assert(curr_delta[0].size() == layer_->out_shape()[0].size());

    backward_activation(*out_grad[0], *out_data[0], curr_delta);

    fill_tensor(*prev_delta, float_t{0});

    for (size_t i = 0; i < prev_out.size(); i++) {
      kernels::tiny_quantized_conv2d_back_kernel(*params_c_, *prev_out[i], W,
                                                 dW[i], db[i], curr_delta[i],
                                                 &(*prev_delta)[i]);
    }

    if (params_c_->pad_type == padding::same) {
      copy_and_unpad_delta(cws.prev_delta_padded_, *in_grad[0]);
    }
  }

  // quantized deconvolution
  void deconv2d_q(const std::vector<Tensor<> *> &in_data,
                  std::vector<Tensor<> *> &out_data) override {
    (*deconv_layer_worker_storage_).prev_out_ = in_data[0];

    const Tensor<> &in = *in_data[0];  // input
    auto W             = layer_->parameter_at(0).data();
    auto bias          = params_d_->has_bias ? layer_->parameter_at(1).data() : Tensor<>();
    Tensor<> &out      = *out_data[0];

    // deconv2d-kernel requires padded size buffer
    out.resize_axis(params_d_->out.size());
    out.fill(float_t(0));

    for (size_t i = 0; i < in.size(); i++) {
      kernels::tiny_quantized_deconv2d_kernel(*params_d_, in[i], W, bias,
                                              out[i], layer_->parallelize());
    }

    copy_and_unpad_output(out);
    out = *(*deconv_layer_worker_storage_).curr_out_unpadded_;
  }

  // efficient quantization without abundant quantization/dequantization
  void deconv2d_eq(const std::vector<Tensor<> *> &in_data,
                   std::vector<Tensor<> *> &out_data) override {
    (*deconv_layer_worker_storage_).prev_out_ = in_data[0];

    const Tensor<> &in = *in_data[0];  // input
    auto W             = layer_->parameter_at(0).data();
    auto bias          = params_d_->has_bias ? layer_->parameter_at(1).data() : Tensor<>();
    const Tensor<> &in_r = *in_data[1];
    auto W_r        = layer_->parameter_at(params_d_->has_bias ?2:1).data();
    auto b_r        = params_d_->has_bias ? layer_->parameter_at(3).data() :  Tensor<>();
    Tensor<> &out   = *out_data[0];
    Tensor<> &out_r = *out_data[1];

    // deconv2d-kernel requires padded size buffer
    fill_tensor(out, float_t{0}, params_d_->out.size());

    for (size_t i = 0; i < in.size(); i++) {
      kernels::tiny_quantized_deconv2d_kernel(*params_d_, in[i], W, bias,
                                              in_r[i], W_r, b_r, out[i],
                                              out_r[i], layer_->parallelize());
    }

    copy_and_unpad_output(out);
    out = *(*deconv_layer_worker_storage_).curr_out_unpadded_;
  }

  void deconv2d_q(const std::vector<Tensor<> *> &in_data,
                  const std::vector<Tensor<> *> &out_data,
                  std::vector<Tensor<> *> &out_grad,
                  std::vector<Tensor<> *> &in_grad) override {
    deconv_layer_worker_specific_storage &cws = (*deconv_layer_worker_storage_);
    if (params_d_->pad_type == padding::same)
      copy_and_pad_delta(cws.curr_delta_padded, *in_grad[0]);

    const Tensor<> &prev_out = *(cws.prev_out_);

    auto W       = layer_->parameter_at(0).data();
    Tensor<> &dW = layer_->parameter_at(0).grad()-
    Tensor<> &db = layer_->parameter_at(1).grad();
    Tensor<> &curr_delta = (params_d_->pad_type == padding::same)
                             ? cws.curr_delta_padded
                             : *out_grad[1];
    Tensor<> *prev_delta = in_grad[0];

    assert(W.size() == params_d_->weight.size());
    assert(dW[0].size() == params_d_->weight.size());
    assert(curr_delta[0].size() == layer_->out_shape()[0].size());

    backward_activation(*out_grad[0], *out_data[0], curr_delta);

    prev_delta->fill(float_t(0.0));

    for (size_t i = 0; i < prev_out.size(); i++) {
      kernels::tiny_quantized_deconv2d_back_kernel(*params_d_, prev_out[i], W,
                                                   dW[i], db[i], curr_delta[i],
                                                   &(*prev_delta)[i]);
    }
  }

  void fully_q(const std::vector<Tensor<> *> &in_data,
               std::vector<Tensor<> *> &out_data) override {
#ifdef CNN_USE_GEMMLOWP
    const Tensor<> &in = *in_data[0];
    auto W             = layer_->parameter_at(0).data();
    Tensor<> &out = *out_data[0];

    for (size_t i = 0; i < in.size(); i++) {
      kernels::tiny_quantized_fully_connected_kernel(
        *params_f_, in[i], W, params_f_->has_bias_ ? (*in_data[2])[0] : vec_t(),
        out[i], layer_->parallelize());
    }
#else
    CNN_UNREFERENCED_PARAMETER(in_data);
    CNN_UNREFERENCED_PARAMETER(out_data);
    throw nn_not_implemented_error(
      "quantized fully op requires gemmlowp "
      "library. please define CNN_USE_GEMMLOWP");
#endif
  }

  void fully_eq(const std::vector<Tensor<> *> &in_data,
                std::vector<Tensor<> *> &out_data) override {
#ifdef CNN_USE_GEMMLOWP
    const Tensor<> &in = *in_data[0];
    auto W             = layer_->parameter_at(0).data();
    auto b             = params_f_->has_bias_ ? layer_->parameter_at(1).data() : Tensor<>();
    const Tensor<> &in_r = *in_data[1];
    auto W_r        = layer_->parameter_at(params_f_->has_bias_?2:1).data();
    auto b_r        = params_f_->has_bias_ ? layer_->parameter_at(3).data(): Tensor<>();
    Tensor<> &out   = *out_data[0];
    Tensor<> &out_r = *out_data[1];

    for (size_t i = 0; i < in.size(); i++) {
      kernels::tiny_quantized_fully_connected_kernel(
        *params_f_, in[i], W, b, in_r[i], W_r, b_r, out[i], out_r[i],
        layer_->parallelize());
    }
#else
    CNN_UNREFERENCED_PARAMETER(in_data);
    CNN_UNREFERENCED_PARAMETER(out_data);
    throw nn_not_implemented_error(
      "quantized fully op requires gemmlowp "
      "library. please define CNN_USE_GEMMLOWP");
#endif
  }

  void fully_q(const std::vector<Tensor<> *> &in_data,
               const std::vector<Tensor<> *> &out_data,
               std::vector<Tensor<> *> &out_grad,
               std::vector<Tensor<> *> &in_grad) override {
#ifdef CNN_USE_GEMMLOWP
    const Tensor<> &prev_out = *in_data[0];

    auto W       = layer_->parameter_at(0).data();
    Tensor<> &dW = layer_->parameter_at(0).grad();
    Tensor<> &db = layer_->parameter_at(1).grad();
    Tensor<> &prev_delta = *in_grad[0];
    Tensor<> &curr_delta = *out_grad[0];

    backward_activation(*out_grad[0], *out_data[0], curr_delta);

    for (size_t i = 0; i < prev_out.size(); i++) {
      kernels::tiny_quantized_fully_connected_back_kernel(
        *params_f_, prev_out[i], W, dW[i], prev_delta[i], curr_delta[i], db[i],
        layer_->parallelize());
    }
#else
    CNN_UNREFERENCED_PARAMETER(in_data);
    CNN_UNREFERENCED_PARAMETER(out_data);
    CNN_UNREFERENCED_PARAMETER(out_grad);
    CNN_UNREFERENCED_PARAMETER(in_grad);
    throw nn_not_implemented_error(
      "quantized fully op requires gemmlowp "
      "library. please define CNN_USE_GEMMLOWP");
#endif
  }

  backend_t type() const override { return default_engine(); }

 private:
  /* Pointer to the convolution parameters */
  conv_params *params_c_;
  deconv_params *params_d_;
  // fully_params *params_f_;

  /* Pointer to the workers */
  conv_layer_worker_specific_storage *conv_layer_worker_storage_;
  deconv_layer_worker_specific_storage *deconv_layer_worker_storage_;

  /* Pointers to parent class functions */
  std::function<void(const Tensor<> &)> copy_and_pad_input;
  std::function<void(const Tensor<> &)> copy_and_unpad_output;
  std::function<void(const Tensor<> &, Tensor<> &)> copy_and_unpad_delta;
  std::function<void(const Tensor<> &, Tensor<> &)> copy_and_pad_delta;
  std::function<void(const Tensor<> &, const Tensor<> &, Tensor<> &)>
    backward_activation;
};

}  // namespace core
}  // namespace tiny_dnn
