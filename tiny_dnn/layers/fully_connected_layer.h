/*
    Copyright (c) 2013, Taiga Nomi
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
template <typename Activation>
class fully_connected_layer : public feedforward_layer<Activation> {
 public:
  typedef feedforward_layer<Activation> Base;
  CNN_USE_LAYER_MEMBERS;

  /**
   * @param in_dim [in] number of elements of the input
   * @param out_dim [in] number of elements of the output
   * @param has_bias [in] whether to include additional bias to the layer
   **/
  fully_connected_layer(serial_size_t in_dim,
                        serial_size_t out_dim,
                        bool has_bias          = true,
                        backend_t backend_type = core::default_engine())
    : Base(std_input_order(has_bias)) {
    set_params(in_dim, out_dim, has_bias);
    init_backend(backend_type);
    Base::set_backend_type(backend_type);
  }

  // move constructor
  fully_connected_layer(fully_connected_layer &&other)
    : Base(std::move(other)),
      params_(std::move(other.params_)),
      kernel_fwd_(std::move(other.kernel_fwd_)),
      kernel_back_(std::move(other.kernel_back_)) {
    init_backend(std::move(other.engine()));
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
    // forward convolutional op context
    auto ctx = OpKernelContext(in_data, out_data);
    ctx.setParallelize(layer::parallelize());
    ctx.setEngine(layer::engine());

    // launch convolutional kernel
    kernel_fwd_->compute(ctx);

    // activations
    this->forward_activation(*out_data[0], *out_data[1]);
  }

  void back_propagation(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) override {
    // activations
    // TODO(edgar/nyanp): refactor and move activations outside
    this->backward_activation(*out_grad[0], *out_data[0], *out_grad[1]);

    // backward convolutional op context
    auto ctx = OpKernelContext(in_data, out_data, out_grad, in_grad);
    ctx.setParallelize(layer::parallelize());
    ctx.setEngine(layer::engine());

    // launch convolutional kernel
    kernel_back_->compute(ctx);
  }

  std::string layer_type() const override { return "fully-connected"; }

#ifndef CNN_NO_SERIALIZATION
  friend struct serialization_buddy;
#endif

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

  /* Forward and backward ops */
  std::shared_ptr<core::OpKernel> kernel_fwd_;
  std::shared_ptr<core::OpKernel> kernel_back_;
};

}  // namespace tiny_dnn
