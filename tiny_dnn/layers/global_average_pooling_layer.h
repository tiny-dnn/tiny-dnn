/*
    Copyright (c) 2015, Taiga Nomi
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "tiny_dnn/layers/layer.h"
#include "tiny_dnn/util/util.h"

#include "tiny_dnn/core/kernels/global_avepool_grad_op.h"
#include "tiny_dnn/core/kernels/global_avepool_op.h"

namespace tiny_dnn {

/**
 * applies channel-wise global average pooling to spatial data.
 **/
class global_average_pooling_layer : public layer {
 public:
  using layer::parallelize_;

  global_average_pooling_layer(const shape3d &in_shape,
                               backend_t backend_type = core::default_engine())
    : global_average_pooling_layer(
        in_shape.width_, in_shape.height_, in_shape.depth_, backend_type) {}

  /**
   * @param in_width     [in] width of input image
   * @param in_height    [in] height of input image
   * @param in_channels  [in] the number of input image channels (depth)
  **/
  global_average_pooling_layer(serial_size_t in_width,
                               serial_size_t in_height,
                               serial_size_t in_channels,
                               backend_t backend_type = core::default_engine())
    : layer({vector_type::data}, {vector_type::data}) {
    set_global_avepool_params(shape3d(in_width, in_height, in_channels),
                              shape3d(in_channels, 1, 1));

    init_backend(backend_type);
  }

  // move constructor
  global_average_pooling_layer(global_average_pooling_layer &&other)  // NOLINT
    : layer(std::move(other)), params_(std::move(other.params_)) {
    init_backend(std::move(layer::engine()));
  }

  serial_size_t fan_in_size() const override {
    return static_cast<serial_size_t>(params_.in.width_ * params_.in.height_);
  }

  serial_size_t fan_out_size() const override { return 1; }

  void forward_propagation(const std::vector<tensor_t *> &in_data,
                           std::vector<tensor_t *> &out_data) override {
    auto ctx = OpKernelContext(in_data, out_data);
    ctx.setParallelize(layer::parallelize());
    ctx.setEngine(layer::engine());

    kernel_fwd_->compute(ctx);
  }

  void back_propagation(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) override {
    auto ctx = OpKernelContext(in_data, out_data, out_grad, in_grad);
    ctx.setParallelize(layer::parallelize());
    ctx.setEngine(layer::engine());

    kernel_back_->compute(ctx);
  }

  std::vector<index3d<serial_size_t>> in_shape() const override {
    return {params_.in};
  }

  std::vector<index3d<serial_size_t>> out_shape() const override {
    return {params_.out};
  }

  std::string layer_type() const override {
    return std::string("global-ave-pool");
  }

  std::pair<serial_size_t, serial_size_t> pool_size() const {
    return std::make_pair(params_.in.width_, params_.in.height_);
  }

  friend struct serialization_buddy;

 private:
  global_avepool_params params_;

  /* Forward and backward ops */
  std::shared_ptr<core::OpKernel> kernel_fwd_;
  std::shared_ptr<core::OpKernel> kernel_back_;

  void init_backend(backend_t backend_type) {
    core::OpKernelConstruction ctx =
      core::OpKernelConstruction(layer::device(), &params_);

    layer::set_backend_type(backend_type);
    if (backend_type == backend_t::internal || backend_type == backend_t::avx ||
        backend_type == backend_t::nnpack) {
      kernel_fwd_.reset(new GlobalAvePoolOp(ctx));
      kernel_back_.reset(new GlobalAvePoolGradOp(ctx));
      return;
    } else {
      throw nn_error("Not supported engine: " + to_string(backend_type));
    }
  }

  void set_global_avepool_params(const shape3d &in, const shape3d &out) {
    params_.in  = in;
    params_.out = out;
  }
};

}  // namespace tiny_dnn
