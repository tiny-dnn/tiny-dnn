/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "tiny_dnn/layers/partial_connected_layer.h"
#include "tiny_dnn/util/util.h"

#ifdef DNN_USE_IMAGE_API
#include "tiny_dnn/util/image.h"
#endif  // DNN_USE_IMAGE_API

namespace tiny_dnn {

// forward_propagation
inline void tiny_average_pooling_kernel(
  bool parallelize,
  const std::vector<tensor_t *> &in_data,
  std::vector<tensor_t *> &out_data,
  const shape3d &out_dim,
  float_t scale_factor,
  std::vector<typename partial_connected_layer::wi_connections> &out2wi) {
  for_i(parallelize, in_data[0]->size(), [&](size_t sample) {
    const vec_t &in = (*in_data[0])[sample];
    const vec_t &W  = (*in_data[1])[0];
    const vec_t &b  = (*in_data[2])[0];
    vec_t &out      = (*out_data[0])[sample];

    auto oarea = out_dim.area();
    size_t idx = 0;
    for (size_t d = 0; d < out_dim.depth_; ++d) {
      float_t weight = W[d] * scale_factor;
      float_t bias   = b[d];
      for (size_t i = 0; i < oarea; ++i, ++idx) {
        const auto &connections = out2wi[idx];
        float_t value{0};
        for (auto connection : connections) value += in[connection.second];
        value *= weight;
        value += bias;
        out[idx] = value;
      }
    }

    assert(out.size() == out2wi.size());
  });
}

// back_propagation
inline void tiny_average_pooling_back_kernel(
  bool parallelize,
  const std::vector<tensor_t *> &in_data,
  const std::vector<tensor_t *> &out_data,
  std::vector<tensor_t *> &out_grad,
  std::vector<tensor_t *> &in_grad,
  const shape3d &in_dim,
  float_t scale_factor,
  std::vector<typename partial_connected_layer::io_connections> &weight2io,
  std::vector<typename partial_connected_layer::wo_connections> &in2wo,
  std::vector<std::vector<size_t>> &bias2out) {
  CNN_UNREFERENCED_PARAMETER(out_data);
  for_i(parallelize, in_data[0]->size(), [&](size_t sample) {
    const vec_t &prev_out = (*in_data[0])[sample];
    const vec_t &W        = (*in_data[1])[0];
    vec_t &dW             = (*in_grad[1])[sample];
    vec_t &db             = (*in_grad[2])[sample];
    vec_t &prev_delta     = (*in_grad[0])[sample];
    vec_t &curr_delta     = (*out_grad[0])[sample];

    auto inarea = in_dim.area();
    size_t idx  = 0;
    for (size_t i = 0; i < in_dim.depth_; ++i) {
      float_t weight = W[i] * scale_factor;
      for (size_t j = 0; j < inarea; ++j, ++idx) {
        prev_delta[idx] = weight * curr_delta[in2wo[idx][0].second];
      }
    }

    for (size_t i = 0; i < weight2io.size(); ++i) {
      const auto &connections = weight2io[i];
      float_t diff{0};

      for (auto connection : connections)
        diff += prev_out[connection.first] * curr_delta[connection.second];

      dW[i] += diff * scale_factor;
    }

    for (size_t i = 0; i < bias2out.size(); i++) {
      const std::vector<size_t> &outs = bias2out[i];
      float_t diff{0};

      for (auto o : outs) diff += curr_delta[o];

      db[i] += diff;
    }
  });
}

/**
 * average pooling with trainable weights
 **/
class average_pooling_layer : public partial_connected_layer {
 public:
  using Base = partial_connected_layer;

  /**
   * @param in_width     [in] width of input image
   * @param in_height    [in] height of input image
   * @param in_channels  [in] the number of input image channels(depth)
   * @param pool_size    [in] factor by which to downscale
   **/
  average_pooling_layer(size_t in_width,
                        size_t in_height,
                        size_t in_channels,
                        size_t pool_size)
    : average_pooling_layer(in_width,
                            in_height,
                            in_channels,
                            pool_size,
                            (in_height == 1 ? 1 : pool_size)) {}

  average_pooling_layer(const shape3d &in_shape,
                        size_t pool_size,
                        size_t stride)
    : average_pooling_layer(
        in_shape.width_, in_shape.width_, in_shape.depth_, pool_size, stride) {}

  /**
   * @param in_width     [in] width of input image
   * @param in_height    [in] height of input image
   * @param in_channels  [in] the number of input image channels(depth)
   * @param pool_size    [in] factor by which to downscale
   * @param stride       [in] interval at which to apply the filters to the
   *input
  **/
  average_pooling_layer(size_t in_width,
                        size_t in_height,
                        size_t in_channels,
                        size_t pool_size,
                        size_t stride)
    : average_pooling_layer(in_width,
                            in_height,
                            in_channels,
                            pool_size,
                            (in_height == 1 ? 1 : pool_size),
                            stride,
                            stride,
                            padding::valid) {}

  /**
   * @param in_width     [in] width of input image
   * @param in_height    [in] height of input image
   * @param in_channels  [in] the number of input image channels(depth)
   * @param pool_size_x  [in] factor by which to downscale
   * @param pool_size_y  [in] factor by which to downscale
   * @param stride_x     [in] interval at which to apply the filters to the
   *input
   * @param stride_y     [in] interval at which to apply the filters to the
   *input
   * @param pad_type     [in] padding mode(same/valid)
   **/
  average_pooling_layer(size_t in_width,
                        size_t in_height,
                        size_t in_channels,
                        size_t pool_size_x,
                        size_t pool_size_y,
                        size_t stride_x,
                        size_t stride_y,
                        padding pad_type = padding::valid)
    : Base(in_width * in_height * in_channels,
           conv_out_length(in_width, pool_size_x, stride_x, pad_type) *
             conv_out_length(in_height, pool_size_y, stride_y, pad_type) *
             in_channels,
           in_channels,
           in_channels,
           float_t(1) / (pool_size_x * pool_size_y)),
      stride_x_(stride_x),
      stride_y_(stride_y),
      pool_size_x_(pool_size_x),
      pool_size_y_(pool_size_y),
      pad_type_(pad_type),
      in_(in_width, in_height, in_channels),
      out_(conv_out_length(in_width, pool_size_x, stride_x, pad_type),
           conv_out_length(in_height, pool_size_y, stride_y, pad_type),
           in_channels),
      w_(pool_size_x, pool_size_y, in_channels) {
    if ((in_width % pool_size_x) || (in_height % pool_size_y)) {
      pooling_size_mismatch(in_width, in_height, pool_size_x, pool_size_y);
    }

    init_connection(pool_size_x, pool_size_y);
  }

  std::vector<index3d<size_t>> in_shape() const override {
    return {in_, w_, index3d<size_t>(1, 1, out_.depth_)};
  }

  std::vector<index3d<size_t>> out_shape() const override { return {out_}; }

  std::string layer_type() const override { return "ave-pool"; }

  void forward_propagation(const std::vector<tensor_t *> &in_data,
                           std::vector<tensor_t *> &out_data) override {
    tiny_average_pooling_kernel(parallelize_, in_data, out_data, out_,
                                Base::scale_factor_, Base::out2wi_);
  }

  void back_propagation(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) override {
    tiny_average_pooling_back_kernel(
      parallelize_, in_data, out_data, out_grad, in_grad, in_,
      Base::scale_factor_, Base::weight2io_, Base::in2wo_, Base::bias2out_);
  }

  std::pair<size_t, size_t> pool_size() const {
    return std::make_pair(pool_size_x_, pool_size_y_);
  }

  friend struct serialization_buddy;

 private:
  size_t stride_x_;
  size_t stride_y_;
  size_t pool_size_x_;
  size_t pool_size_y_;
  padding pad_type_;
  shape3d in_;
  shape3d out_;
  shape3d w_;

  static size_t pool_out_dim(size_t in_size,
                             size_t pooling_size,
                             size_t stride) {
    return static_cast<int>(
      std::ceil((static_cast<float_t>(in_size) - pooling_size) / stride) + 1);
  }

  void init_connection(size_t pooling_size_x, size_t pooling_size_y) {
    for (size_t c = 0; c < in_.depth_; ++c) {
      for (size_t y = 0; y < in_.height_ - pooling_size_y + 1; y += stride_y_) {
        for (size_t x = 0; x < in_.width_ - pooling_size_x + 1;
             x += stride_x_) {
          connect_kernel(pooling_size_x, pooling_size_y, x, y, c);
        }
      }
    }

    for (size_t c = 0; c < in_.depth_; ++c) {
      for (size_t y = 0; y < out_.height_; ++y) {
        for (size_t x = 0; x < out_.width_; ++x) {
          this->connect_bias(c, out_.get_index(x, y, c));
        }
      }
    }
  }

  void connect_kernel(size_t pooling_size_x,
                      size_t pooling_size_y,
                      size_t x,
                      size_t y,
                      size_t inc) {
    size_t dymax  = std::min(pooling_size_y, in_.height_ - y);
    size_t dxmax  = std::min(pooling_size_x, in_.width_ - x);
    size_t dstx   = x / stride_x_;
    size_t dsty   = y / stride_y_;
    size_t outidx = out_.get_index(dstx, dsty, inc);
    for (size_t dy = 0; dy < dymax; ++dy) {
      for (size_t dx = 0; dx < dxmax; ++dx) {
        this->connect_weight(in_.get_index(x + dx, y + dy, inc), outidx, inc);
      }
    }
  }
};

}  // namespace tiny_dnn
