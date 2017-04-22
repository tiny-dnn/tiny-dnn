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

#include "tiny_dnn/core/kernels/maxpool_grad_op.h"
#include "tiny_dnn/core/kernels/maxpool_op.h"

#include "tiny_dnn/util/util.h"

#ifdef DNN_USE_IMAGE_API
#include "tiny_dnn/util/image.h"
#endif

namespace tiny_dnn {

/**
 * applies max-pooing operaton to the spatial data
 **/
class max_pooling_layer : public layer {
 public:
  using layer::parallelize_;

  /**
   * @param in_width     [in] width of input image
   * @param in_height    [in] height of input image
   * @param in_channels  [in] the number of input image channels(depth)
   * @param pooling_size [in] factor by which to downscale
   **/
  max_pooling_layer(serial_size_t in_width,
                    serial_size_t in_height,
                    serial_size_t in_channels,
                    serial_size_t pooling_size,
                    backend_t backend_type = core::default_engine())
    : max_pooling_layer(in_width,
                        in_height,
                        in_channels,
                        pooling_size,
                        (in_height == 1 ? 1 : pooling_size),
                        backend_type) {}

  max_pooling_layer(const shape3d &in_shape,
                    serial_size_t pooling_size,
                    serial_size_t stride,
                    backend_t backend_type = core::default_engine())
    : max_pooling_layer(in_shape.width_,
                        in_shape.height_,
                        in_shape.depth_,
                        pooling_size,
                        stride,
                        backend_type) {}

  max_pooling_layer(serial_size_t in_width,
                    serial_size_t in_height,
                    serial_size_t in_channels,
                    serial_size_t pooling_size,
                    serial_size_t stride,
                    backend_t backend_type = core::default_engine())
    : max_pooling_layer(in_width,
                        in_height,
                        in_channels,
                        pooling_size,
                        (in_height == 1 ? 1 : pooling_size),
                        stride,
                        stride,
                        padding::valid,
                        backend_type) {}

  /**
   * @param in_width     [in] width of input image
   * @param in_height    [in] height of input image
   * @param in_channels  [in] the number of input image channels(depth)
   * @param pooling_size [in] factor by which to downscale
   * @param stride       [in] interval at which to apply the filters to the
   *input
  **/
  max_pooling_layer(serial_size_t in_width,
                    serial_size_t in_height,
                    serial_size_t in_channels,
                    serial_size_t pooling_size_x,
                    serial_size_t pooling_size_y,
                    serial_size_t stride_x,
                    serial_size_t stride_y,
                    padding pad_type       = padding::valid,
                    backend_t backend_type = core::default_engine())
    : layer({vector_type::data}, {vector_type::data}) {
    set_maxpool_params(
      shape3d(in_width, in_height, in_channels),
      shape3d(conv_out_length(in_width, pooling_size_x, stride_x, pad_type),
              conv_out_length(in_height, pooling_size_y, stride_y, pad_type),
              in_channels),
      pooling_size_x, pooling_size_y, stride_x, stride_y, pad_type);

    init_connection();
    init_backend(backend_type);
    layer::set_backend_type(backend_type);
  }

  // move constructor
  max_pooling_layer(max_pooling_layer &&other)  // NOLINT
    : layer(std::move(other)), params_(std::move(other.params_)) {
    init_connection();
    init_backend(std::move(layer::engine()));
  }

  serial_size_t fan_in_size() const override {
    return static_cast<serial_size_t>(params_.out2in[0].size());
  }

  serial_size_t fan_out_size() const override { return 1; }

  void forward_propagation(const std::vector<tensor_t *> &in_data,
                           std::vector<tensor_t *> &out_data) override {
    // forward convolutional op context
    auto ctx = OpKernelContext(in_data, out_data);
    ctx.setParallelize(layer::parallelize());
    ctx.setEngine(layer::engine());

    // launch convolutional kernel
    kernel_fwd_->compute(ctx);
  }

  void back_propagation(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) override {
    // backward convolutional op context
    auto ctx = OpKernelContext(in_data, out_data, out_grad, in_grad);
    ctx.setParallelize(layer::parallelize());
    ctx.setEngine(layer::engine());

    // launch convolutional kernel
    kernel_back_->compute(ctx);
  }

  std::vector<index3d<serial_size_t>> in_shape() const override {
    return {params_.in};
  }

  std::vector<index3d<serial_size_t>> out_shape() const override {
    return {params_.out};
  }

  std::string layer_type() const override { return std::string("max-pool"); }

  std::string kernel_file() const override {
    return std::string("../tiny_cnn/core/kernels/cl_kernels/pooling.cl");
  }

  std::pair<serial_size_t, serial_size_t> pool_size() const {
    return std::make_pair(params_.pool_size_x, params_.pool_size_y);
  }

  void set_sample_count(serial_size_t sample_count) override {
    layer::set_sample_count(sample_count);
    params_.out2inmax.resize(sample_count,
                             std::vector<serial_size_t>(params_.out.size()));
  }

  friend struct serialization_buddy;

 private:
  /* The Max Poling operation params */
  maxpool_params params_;

  /* Forward and backward ops */
  std::shared_ptr<core::OpKernel> kernel_fwd_;
  std::shared_ptr<core::OpKernel> kernel_back_;

  void connect_kernel(serial_size_t pooling_size_x,
                      serial_size_t pooling_size_y,
                      serial_size_t outx,
                      serial_size_t outy,
                      serial_size_t c) {
    serial_size_t dxmax = static_cast<serial_size_t>(
      std::min(static_cast<serial_size_t>(pooling_size_x),
               params_.in.width_ - outx * params_.stride_x));

    serial_size_t dymax = static_cast<serial_size_t>(
      std::min(static_cast<serial_size_t>(pooling_size_y),
               params_.in.height_ - outy * params_.stride_y));

    for (serial_size_t dy = 0; dy < dymax; dy++) {
      for (serial_size_t dx = 0; dx < dxmax; dx++) {
        serial_size_t in_index = params_.in.get_index(
          static_cast<serial_size_t>(outx * params_.stride_x + dx),
          static_cast<serial_size_t>(outy * params_.stride_y + dy), c);
        serial_size_t out_index = params_.out.get_index(outx, outy, c);

        if (in_index >= params_.in2out.size()) {
          throw nn_error("index overflow");
        }
        if (out_index >= params_.out2in.size()) {
          throw nn_error("index overflow");
        }
        params_.in2out[in_index] = out_index;
        params_.out2in[out_index].push_back(in_index);
      }
    }
  }

  void init_connection() {
    params_.in2out.resize(params_.in.size());
    params_.out2in.resize(params_.out.size());

    for (serial_size_t c = 0; c < params_.in.depth_; ++c) {
      for (serial_size_t y = 0; y < params_.out.height_; ++y) {
        for (serial_size_t x = 0; x < params_.out.width_; ++x) {
          connect_kernel(params_.pool_size_x, params_.pool_size_y, x, y, c);
        }
      }
    }
  }

  void init_backend(backend_t backend_type) {
    core::OpKernelConstruction ctx =
      core::OpKernelConstruction(layer::device(), &params_);

    if (backend_type == backend_t::internal ||
        backend_type == backend_t::nnpack || backend_type == backend_t::avx) {
      kernel_fwd_.reset(new MaxPoolOp(ctx));
      kernel_back_.reset(new MaxPoolGradOp(ctx));
      return;
    } else {
      throw nn_error("Not supported engine: " + to_string(backend_type));
    }
  }

  void set_maxpool_params(const shape3d &in,
                          const shape3d &out,
                          serial_size_t pooling_size_x,
                          serial_size_t pooling_size_y,
                          serial_size_t stride_x,
                          serial_size_t stride_y,
                          padding pad_type) {
    params_.in          = in;
    params_.out         = out;
    params_.pool_size_x = pooling_size_x;
    params_.pool_size_y = pooling_size_y;
    params_.stride_x    = stride_x;
    params_.stride_y    = stride_y;
    params_.pad_type    = pad_type;
  }
};

}  // namespace tiny_dnn
