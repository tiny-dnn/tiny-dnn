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

#include "tiny_dnn/core/framework/tensor.h"
#include "tiny_dnn/core/params/avepool_params.h"
#include "tiny_dnn/util/util.h"

#ifdef DNN_USE_IMAGE_API
#include "tiny_dnn/util/image.h"
#endif  // DNN_USE_IMAGE_API

namespace tiny_dnn {

// forward_propagation
inline void tiny_average_pooling_kernel(
  const Tensor<> &in_data,
  const Tensor<> &weights,
  const Tensor<> &biases,
  Tensor<> &out_data,
  const core::avepool_params &params,
  const core::average_pooling_layer_worker_specific_storage &aws,
  const bool layer_parallelize) {
  out_data.reshape({in_data.shape()[0], params.out.area() * params.out.depth_});
  for_i(layer_parallelize, in_data.shape()[0], [&](size_t sample) {
    auto oarea = params.out.area();
    size_t idx = 0;

    for (size_t d = 0; d < params.out.depth_; ++d) {
      float_t weight = weights.host_at(d) * params.scale_factor;
      float_t bias   = biases.host_at(d);
      for (size_t i = 0; i < oarea; ++i, ++idx) {
        const auto &connections = aws.out2wi[idx];
        float_t value{0};
        for (auto connection : connections)
          value += in_data.host_at(sample, connection.second);
        value *= weight;
        value += bias;
        out_data.host_at(sample, idx) = value;
      }
    }
  });
}

// back_propagation
inline void tiny_average_pooling_back_kernel(
  const Tensor<> &prev_out,
  const Tensor<> &weights,
  Tensor<> &weights_grads,
  Tensor<> &bias_grads,
  Tensor<> &curr_delta,
  Tensor<> &prev_delta,
  const core::avepool_params &params,
  const core::average_pooling_layer_worker_specific_storage &aws,
  const bool layer_parallelize) {
  prev_delta.reshape(
    {prev_out.shape()[0], params.in.area() * params.in.depth_});
  weights_grads.reshape(
    {prev_out.shape()[0], params.in.area() * params.in.depth_});
  bias_grads.reshape(
    {prev_out.shape()[0], params.in.area() * params.in.depth_});
  for_i(layer_parallelize, prev_out.shape()[0], [&](size_t sample) {
    auto inarea = params.in.area();
    size_t idx  = 0;
    for (size_t i = 0; i < params.in.depth_; ++i) {
      float_t weight = weights.host_at(i) * params.scale_factor;
      for (size_t j = 0; j < inarea; ++j, ++idx) {
        prev_delta.host_at(sample, idx) =
          weight * curr_delta.host_at(sample, aws.in2wo[idx][0].second);
      }
    }

    for (size_t i = 0; i < aws.weight2io.size(); ++i) {
      const auto &connections = aws.weight2io[i];
      float_t diff{0};

      for (auto connection : connections)
        diff += prev_out.host_at(sample, connection.first) *
                curr_delta.host_at(sample, connection.second);

      weights_grads.host_at(sample, i) += diff * params.scale_factor;
    }

    for (size_t i = 0; i < aws.bias2out.size(); i++) {
      const std::vector<size_t> &outs = aws.bias2out[i];
      float_t diff{0};

      for (auto o : outs) diff += curr_delta.host_at(sample, o);

      bias_grads.host_at(sample, i) += diff;
    }
  });
}

/**
 * average pooling with trainable weights
 **/
class average_pooling_layer : public layer {
 public:
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
    : layer({vector_type::data}, {vector_type::data}) {
    avepool_set_params(in_width, in_height, in_channels, pool_size_x,
                       pool_size_y, stride_x, stride_y, pad_type);
    aws_ = core::average_pooling_layer_worker_specific_storage(
      in_channels, params_.out.size(), params_.in.size(), in_channels,
      params_.out.size());

    // todo (karandesai) : add has_bias flag and necessary consequences
    layer::add_parameter(in_channels, in_channels, in_height, in_width,
                         parameter_type::weight);
    layer::add_parameter(1, 1, 1, in_channels, parameter_type::bias);

    if ((in_width % pool_size_x) || (in_height % pool_size_y)) {
      pooling_size_mismatch(in_width, in_height, pool_size_x, pool_size_y);
    }

    init_connection(pool_size_x, pool_size_y);
  }

  std::vector<shape3d> in_shape() const override { return {params_.in}; }

  std::vector<shape3d> out_shape() const override { return {params_.out}; }

  std::string layer_type() const override { return "ave-pool"; }

  void forward_propagation(const std::vector<Tensor<> *> &in_data,
                           std::vector<Tensor<> *> &out_data) override {
    // todo (karandesai) : transfer all this into OpKernel
    // OpKernels do not accept worker storage so currently tricky to do so
    out_data[0]->fill(0);

    const Tensor<> weights = *(layer::parameter_at(0).data());
    const Tensor<> bias    = *(layer::parameter_at(1).data());

    tiny_average_pooling_kernel(*in_data[0], weights, bias, *out_data[0],
                                params_, aws_, parallelize_);
  }

  void back_propagation(const std::vector<Tensor<> *> &in_data,
                        const std::vector<Tensor<> *> &out_data,
                        std::vector<Tensor<> *> &out_grad,
                        std::vector<Tensor<> *> &in_grad) override {
    in_grad[0]->fill(0);
    // todo (karandesai) : transfer all this into OpKernel

    const Tensor<> weights = *(layer::parameter_at(0).data());
    const Tensor<> bias    = *(layer::parameter_at(1).data());
    Tensor<> weights_grads = *(layer::parameter_at(0).grad());
    Tensor<> bias_grads    = *(layer::parameter_at(1).grad());

    tiny_average_pooling_back_kernel(*in_data[0], weights, weights_grads,
                                     bias_grads, *out_grad[0], *in_grad[0],
                                     params_, aws_, parallelize_);
    // TODO(Randl): pass by reference?
    layer::parameter_at(0).set_data(weights_grads);
    layer::parameter_at(1).set_data(bias_grads);
  }

  std::pair<size_t, size_t> pool_size() const {
    return std::make_pair(params_.window.width_, params_.window.height_);
  }

  friend struct serialization_buddy;

 private:
  core::avepool_params params_;
  core::average_pooling_layer_worker_specific_storage aws_;

  void avepool_set_params(size_t in_width,
                          size_t in_height,
                          size_t in_channels,
                          size_t pool_size_x,
                          size_t pool_size_y,
                          size_t stride_x,
                          size_t stride_y,
                          padding pad_type = padding::valid) {
    params_.in = shape3d(in_width, in_height, in_channels);
    params_.out =
      pad_type == padding::same
        ? params_.in
        : shape3d(pool_out_dim(in_width, pool_size_x, stride_x),
                  pool_out_dim(in_height, pool_size_y, stride_y), in_channels);
    params_.window       = shape3d(pool_size_x, pool_size_y, in_channels);
    params_.stride_x     = stride_x;
    params_.stride_y     = stride_y;
    params_.pad_type     = pad_type;
    params_.scale_factor = 1.0 / (pool_size_x * pool_size_y);
  }

  static size_t pool_out_dim(size_t in_size,
                             size_t pooling_size,
                             size_t stride) {
    return static_cast<int>(
      std::ceil((static_cast<float_t>(in_size) - pooling_size) / stride) + 1);
  }

  void init_connection(size_t pooling_size_x, size_t pooling_size_y) {
    for (size_t c = 0; c < params_.in.depth_; ++c) {
      for (size_t y = 0; y < params_.in.height_ - pooling_size_y + 1;
           y += params_.stride_y) {
        for (size_t x = 0; x < params_.in.width_ - pooling_size_x + 1;
             x += params_.stride_x) {
          connect_kernel(pooling_size_x, pooling_size_y, x, y, c);
        }
      }
    }

    for (size_t c = 0; c < params_.in.depth_; ++c) {
      for (size_t y = 0; y < params_.out.height_; ++y) {
        for (size_t x = 0; x < params_.out.width_; ++x) {
          this->connect_bias(c, params_.out.get_index(x, y, c));
        }
      }
    }
  }

  void connect_kernel(size_t pooling_size_x,
                      size_t pooling_size_y,
                      size_t x,
                      size_t y,
                      size_t inc) {
    size_t dymax  = std::min(pooling_size_y, params_.in.height_ - y);
    size_t dxmax  = std::min(pooling_size_x, params_.in.width_ - x);
    size_t dstx   = x / params_.stride_x;
    size_t dsty   = y / params_.stride_y;
    size_t outidx = params_.out.get_index(dstx, dsty, inc);
    for (size_t dy = 0; dy < dymax; ++dy) {
      for (size_t dx = 0; dx < dxmax; ++dx) {
        this->connect_weight(params_.in.get_index(x + dx, y + dy, inc), outidx,
                             inc);
      }
    }
  }

  void connect_weight(size_t input_index,
                      size_t output_index,
                      size_t weight_index) {
    aws_.weight2io[weight_index].emplace_back(input_index, output_index);
    aws_.out2wi[output_index].emplace_back(weight_index, input_index);
    aws_.in2wo[input_index].emplace_back(weight_index, output_index);
  }

  void connect_bias(size_t bias_index, size_t output_index) {
    aws_.out2bias[output_index] = bias_index;
    aws_.bias2out[bias_index].push_back(output_index);
  }
};

}  // namespace tiny_dnn
