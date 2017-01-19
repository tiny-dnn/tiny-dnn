/*
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/layers/layer.h"
#include "tiny_dnn/util/product.h"

namespace tiny_dnn {

/**
 * compute fully-connected(matmul) operation
 **/
template <typename Activation>
class quantized_fully_connected_layer : public feedforward_layer<Activation> {
 public:
  typedef feedforward_layer<Activation> Base;
  CNN_USE_LAYER_MEMBERS;

  /**
   * @param in_dim [in] number of elements of the input
   * @param out_dim [in] number of elements of the output
   * @param has_bias [in] whether to include additional bias to the layer
   **/
  quantized_fully_connected_layer(
    serial_size_t in_dim,
    serial_size_t out_dim,
    bool has_bias          = true,
    backend_t backend_type = core::backend_t::internal)
    : Base(std_input_order(has_bias)) {
    set_params(in_dim, out_dim, has_bias);
    init_backend(backend_type);
  }

  // move constructor
  quantized_fully_connected_layer(quantized_fully_connected_layer &&other)
    : Base(std::move(other)), params_(std::move(other.params_)) {
    init_backend(core::backend_t::internal);
  }

  serial_size_t fan_in_size() const override { return params_.in_size_; }

  serial_size_t fan_out_size() const override { return params_.out_size_; }

  std::vector<index3d<serial_size_t>> in_shape() const override {
    if (params_.has_bias_) {
      return {index3d<serial_size_t>(params_.in_size_, 1, 1),
              index3d<serial_size_t>(params_.in_size_, params_.out_size_, 1),
              index3d<serial_size_t>(params_.out_size_, 1, 1)};
    } else {
      return {index3d<serial_size_t>(params_.in_size_, 1, 1),
              index3d<serial_size_t>(params_.in_size_, params_.out_size_, 1)};
    }
  }

  std::vector<index3d<serial_size_t>> out_shape() const override {
    return {index3d<serial_size_t>(params_.out_size_, 1, 1),
            index3d<serial_size_t>(params_.out_size_, 1, 1)};
  }

  void forward_propagation(const std::vector<tensor_t *> &in_data,
                           std::vector<tensor_t *> &out_data) override {
    if (in_data.size() == 2 || in_data.size() == 3) {
      Base::backend_->fully_q(in_data, out_data);

      // activations
      this->forward_activation(*out_data[0], *out_data[1]);
    } else if (in_data.size() == 4 || in_data.size() == 6) {
      Base::backend_->fully_eq(in_data, out_data);
    }
  }

  void back_propagation(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) override {
    Base::backend_->fully_q(in_data, out_data, out_grad, in_grad);
  }

  std::string layer_type() const override { return "q_fully-connected"; }

 protected:
  fully_params params_;

  void set_params(const serial_size_t in_size,
                  const serial_size_t out_size,
                  bool has_bias) {
    params_.in_size_  = in_size;
    params_.out_size_ = out_size;
    params_.has_bias_ = has_bias;
  }

  void init_backend(backend_t backend_type) {
    std::shared_ptr<core::backend> backend = nullptr;

    // allocate new backend
    if (backend_type == backend_t::internal) {
      backend = std::make_shared<core::tiny_backend>(
        &params_, [this](const tensor_t &p_delta, const tensor_t &out,
                         tensor_t &c_delta) {
          return Base::backward_activation(p_delta, out, c_delta);
        });
    } else {
      throw nn_error("Not supported backend type.");
    }

    if (backend) {
      Base::set_backend(backend);
      Base::set_backend_type(backend_type);
      Base::backend_->set_layer(this);
    } else {
      throw nn_error("Could not allocate the backend.");
    }
  }
};

}  // namespace tiny_dnn
