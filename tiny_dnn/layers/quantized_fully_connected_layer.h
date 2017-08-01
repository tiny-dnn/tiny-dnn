/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tiny_dnn/layers/layer.h"
#include "tiny_dnn/util/product.h"

namespace tiny_dnn {

/**
 * compute fully-connected(matmul) operation
 **/
class quantized_fully_connected_layer : public layer {
 public:
  /**
   * @param in_dim [in] number of elements of the input
   * @param out_dim [in] number of elements of the output
   * @param has_bias [in] whether to include additional bias to the layer
   **/
  quantized_fully_connected_layer(
    size_t in_dim,
    size_t out_dim,
    bool has_bias                = true,
    core::backend_t backend_type = core::backend_t::internal)
    : layer(std_input_order(has_bias), {vector_type::data}) {
    set_params(in_dim, out_dim, has_bias);
    init_backend(backend_type);
  }

  // move constructor
  quantized_fully_connected_layer(quantized_fully_connected_layer &&other)
    : layer(std::move(other)), params_(std::move(other.params_)) {
    init_backend(core::backend_t::internal);
  }

  size_t fan_in_size() const override { return params_.in_size_; }

  size_t fan_out_size() const override { return params_.out_size_; }

  std::vector<index3d<size_t>> in_shape() const override {
    if (params_.has_bias_) {
      return {index3d<size_t>(params_.in_size_, 1, 1),
              index3d<size_t>(params_.in_size_, params_.out_size_, 1),
              index3d<size_t>(params_.out_size_, 1, 1)};
    } else {
      return {index3d<size_t>(params_.in_size_, 1, 1),
              index3d<size_t>(params_.in_size_, params_.out_size_, 1)};
    }
  }

  std::vector<index3d<size_t>> out_shape() const override {
    return {index3d<size_t>(params_.out_size_, 1, 1)};
  }

  void forward_propagation(const std::vector<tensor_t *> &in_data,
                           std::vector<tensor_t *> &out_data) override {
    if (in_data.size() == 2 || in_data.size() == 3) {
      layer::backend_->fully_q(in_data, out_data);

    } else if (in_data.size() == 4 || in_data.size() == 6) {
      layer::backend_->fully_eq(in_data, out_data);
    }
  }

  void back_propagation(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) override {
    layer::backend_->fully_q(in_data, out_data, out_grad, in_grad);
  }

  std::string layer_type() const override { return "q_fully-connected"; }

  friend struct serialization_buddy;

 protected:
  core::fully_params params_;

  void set_params(const size_t in_size, const size_t out_size, bool has_bias) {
    params_.in_size_  = in_size;
    params_.out_size_ = out_size;
    params_.has_bias_ = has_bias;
  }

  void init_backend(core::backend_t backend_type) {
    std::shared_ptr<core::backend> backend = nullptr;

    // allocate new backend
    if (backend_type == core::backend_t::internal) {
      backend = std::make_shared<core::tiny_backend>(&params_);
    } else {
      throw nn_error("Not supported backend type.");
    }

    if (backend) {
      layer::set_backend(backend);
      layer::set_backend_type(backend_type);
      layer::backend_->set_layer(this);
    } else {
      throw nn_error("Could not allocate the backend.");
    }
  }
};

}  // namespace tiny_dnn
