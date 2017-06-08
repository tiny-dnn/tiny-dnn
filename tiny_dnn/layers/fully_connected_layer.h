/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once
#include "tiny_dnn/layers/layer.h"

#include "tiny_dnn/core/kernels/fully_connected_grad_op.h"
#include "tiny_dnn/core/kernels/fully_connected_op.h"

namespace tiny_dnn {

/**
 * compute fully-connected(matmul) operation
 **/
class fully_connected_layer : public layer {
 public:
  /**
   * @param in_dim [in] number of elements of the input
   * @param out_dim [in] number of elements of the output
   * @param has_bias [in] whether to include additional bias to the layer
   **/
  fully_connected_layer(serial_size_t in_dim,
                        serial_size_t out_dim,
                        bool has_bias          = true,
                        backend_t backend_type = core::default_engine())
    : layer({vector_type::data}, {vector_type::data}) {
    set_params(in_dim, out_dim, has_bias);
    init_backend(backend_type);
    layer::set_backend_type(backend_type);
    layer::add_parameter(in_dim, out_dim, 1, 1, param_type::weight);
    if (has_bias) {
      layer::add_parameter(out_dim, 1, 1, 1, param_type::bias);
    }
  }

  // move constructor
  fully_connected_layer(fully_connected_layer &&other)
    : layer(std::move(other)),
      params_(std::move(other.params_)),
      kernel_fwd_(std::move(other.kernel_fwd_)),
      kernel_back_(std::move(other.kernel_back_)) {
    init_backend(std::move(other.engine()));
  }

  serial_size_t fan_in_size() const override { return params_.in_size_; }

  serial_size_t fan_out_size() const override { return params_.out_size_; }

  std::vector<shape3d> in_shape() const override {
    return {shape3d(params_.in_size_, 1, 1)};
  }

  std::vector<index3d<serial_size_t>> out_shape() const override {
    return {shape3d(params_.out_size_, 1, 1)};
  }

  void forward_propagation(const std::vector<tensor_t *> &in_data,
                           std::vector<tensor_t *> &out_data) override {
    std::vector<std::shared_ptr<parameter>> p = layer::get_parameters();
    // forward fully connected op context
    fwd_ctx_.set_in_out(in_data, out_data);
    fwd_ctx_.set_parameters(p);
    fwd_ctx_.setParallelize(layer::parallelize());
    fwd_ctx_.setEngine(layer::engine());

    // launch fully connected kernel
    kernel_fwd_->compute(fwd_ctx_);
  }

  void back_propagation(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) override {
    std::vector<std::shared_ptr<parameter>> p = layer::get_parameters();
    // backward fully connected op context
    bwd_ctx_.set_in_out(in_data, out_data, out_grad, in_grad);
    bwd_ctx_.set_parameters(p);
    bwd_ctx_.setParallelize(layer::parallelize());
    bwd_ctx_.setEngine(layer::engine());

    // launch fully connected kernel
    kernel_back_->compute(bwd_ctx_);
  }

  std::string layer_type() const override { return "fully-connected"; }

  friend struct serialization_buddy;

 protected:
  void set_params(const serial_size_t in_size,
                  const serial_size_t out_size,
                  bool has_bias) {
    params_.in_size_  = in_size;
    params_.out_size_ = out_size;
    params_.has_bias_ = has_bias;
  }

  void init_backend(backend_t backend_type) {
    core::OpKernelConstruction ctx =
      core::OpKernelConstruction(layer::device(), &params_);

    if (backend_type == backend_t::internal || backend_type == backend_t::avx ||
        backend_type == backend_t::nnpack) {
      kernel_fwd_.reset(new FullyConnectedOp(ctx));
      kernel_back_.reset(new FullyConnectedGradOp(ctx));
    } else {
      throw nn_error("Not supported engine: " + to_string(backend_type));
    }
  }

 private:
  /* The layer parameters */
  fully_params params_;

  /* forward op context */
  OpKernelContext fwd_ctx_;

  /* backward op context */
  OpKernelContext bwd_ctx_;

  /* Forward and backward ops */
  std::shared_ptr<core::OpKernel> kernel_fwd_;
  std::shared_ptr<core::OpKernel> kernel_back_;
};

}  // namespace tiny_dnn
