/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "tiny_dnn/core/params/aveunpool_params.h"
#include "tiny_dnn/util/util.h"

#ifdef DNN_USE_IMAGE_API
#include "tiny_dnn/util/image.h"
#endif  // DNN_USE_IMAGE_API

namespace tiny_dnn {

// forward_propagation
inline void tiny_average_unpooling_kernel(
  const Tensor<> &in_data,
  const Tensor<> &weights,
  const Tensor<> &biases,
  Tensor<> &out_data,
  const core::aveunpool_params &params,
  const core::average_unpooling_layer_worker_specific_storage &auws,
  bool parallelize) {
  out_data.reshape({in_data.shape()[0], params.out.area() * params.out.depth_});
  for_i(parallelize, in_data.shape()[0], [&](size_t sample) {
    auto oarea = params.out.area();
    size_t idx = 0;
    for (size_t d = 0; d < params.out.depth_; ++d) {
      float_t weight = weights.host_at(d);  // * scale_factor;
      float_t bias   = biases.host_at(d);
      for (size_t i = 0; i < oarea; ++i, ++idx) {
        const auto &connections = auws.out2wi[idx];
        float_t value{0};
        for (auto connection : connections)
          value += in_data.host_at(sample, connection.second);
        value *= weight;
        value += bias;
        out_data.host_at(sample, idx) = value;
      }
    }
    assert(out_data.shape()[1] == auws.out2wi.size());
  });
}

// back_propagation
inline void tiny_average_unpooling_back_kernel(
  const Tensor<> &prev_out,
  const Tensor<> &weights,
  Tensor<> &weights_grads,
  Tensor<> &bias_grads,
  Tensor<> &curr_delta,
  Tensor<> &prev_delta,
  const core::aveunpool_params &params,
  const core::average_unpooling_layer_worker_specific_storage &auws,
  bool parallelize) {
  for_i(parallelize, curr_delta.size(), [&](size_t sample) {
    auto inarea = params.in.area();
    size_t idx  = 0;
    for (size_t i = 0; i < params.in.depth_; ++i) {
      float_t weight = weights.host_at(i);  // * scale_factor;
      for (size_t j = 0; j < inarea; ++j, ++idx) {
        prev_delta.host_at(sample, idx) =
          weight * curr_delta.host_at(sample, auws.in2wo[idx][0].second);
      }
    }

    for (size_t i = 0; i < auws.weight2io.size(); ++i) {
      const auto &connections = auws.weight2io[i];
      float_t diff            = 0.0;

      for (auto connection : connections)
        diff += prev_out.host_at(sample, connection.first) *
                curr_delta.host_at(sample, connection.second);

      weights_grads.host_at(sample, i) += diff;  // * scale_factor;
    }

    for (size_t i = 0; i < auws.bias2out.size(); i++) {
      const std::vector<size_t> &outs = auws.bias2out[i];
      float_t diff                    = 0.0;

      for (auto o : outs) diff += curr_delta.host_at(sample, o);

      bias_grads.host_at(sample, i) += diff;
    }
  });
}

/**
 * average pooling with trainable weights
 **/
class average_unpooling_layer : public layer {
 public:
  /**
   * @param in_width     [in] width of input image
   * @param in_height    [in] height of input image
   * @param in_channels  [in] the number of input image channels(depth)
   * @param pooling_size [in] factor by which to upscale
   **/
  average_unpooling_layer(size_t in_width,
                          size_t in_height,
                          size_t in_channels,
                          size_t pooling_size)
    : average_unpooling_layer(
        in_width, in_height, in_channels, pooling_size, pooling_size) {}

  /**
   * @param in_width     [in] width of input image
   * @param in_height    [in] height of input image
   * @param in_channels  [in] the number of input image channels(depth)
   * @param pooling_size [in] factor by which to upscale
   * @param stride       [in] interval at which to apply the filters to the
   *input
   **/
  average_unpooling_layer(size_t in_width,
                          size_t in_height,
                          size_t in_channels,
                          size_t pooling_size,
                          size_t stride)
    : layer({vector_type::data}, {vector_type::data}) {
    layer::add_parameter(in_channels, in_channels, in_height, in_width,
                         parameter_type::weight);
    layer::add_parameter(1, 1, 1, in_channels, parameter_type::bias);
    aveunpool_set_params(in_width, in_height, in_channels, pooling_size,
                         stride);
    auws_ = core::average_unpooling_layer_worker_specific_storage(
      in_channels, params_.out.size(), params_.in.size(), in_channels);
    init_connection(pooling_size);
  }

  std::vector<shape3d> in_shape() const override { return {params_.in}; }

  std::vector<shape3d> out_shape() const override { return {params_.out}; }

  std::string layer_type() const override { return "ave-unpool"; }

  void forward_propagation(const std::vector<Tensor<> *> &in_data,
                           std::vector<Tensor<> *> &out_data) override {
    // todo (karandesai) : transfer all this into OpKernel
    // OpKernels do not accept worker storage so currently tricky to do so

    const Tensor<> in = *in_data[0];
    Tensor<> &out     = *out_data[0];
    out.fill(0);

    const Tensor<> weights = *(layer::weights_at()[0]->data());
    const Tensor<> biases  = *(layer::bias_at()[0]->data());

    tiny_average_unpooling_kernel(in, weights, biases, out, params_, auws_,
                                  parallelize());
  }

  void back_propagation(const std::vector<Tensor<> *> &in_data,
                        const std::vector<Tensor<> *> &out_data,
                        std::vector<Tensor<> *> &out_grad,
                        std::vector<Tensor<> *> &in_grad) override {
    // todo (karandesai) : transfer all this into OpKernel

    const Tensor<> prev_out = *in_data[0];
    Tensor<> &prev_delta    = *in_grad[0];
    Tensor<> curr_delta     = *out_grad[0];

    const Tensor<> weights = *(layer::weights_at()[0]->data());
    const Tensor<> bias    = *(layer::bias_at()[0]->data());
    Tensor<> weights_grads = *(layer::weights_at()[0]->grad());
    Tensor<> bias_grads    = *(layer::bias_at()[0]->grad());

    prev_delta.fill(0);

    tiny_average_unpooling_back_kernel(prev_out, weights, weights_grads,
                                       bias_grads, curr_delta, prev_delta,
                                       params_, auws_, parallelize_);
    // TODO(karandesai): remove unnecessary assignments
    layer::weights_at()[0]->set_data(weights_grads);
    layer::bias_at()[0]->set_data(bias_grads);
  }

  friend struct serialization_buddy;

 private:
  core::aveunpool_params params_;
  core::average_unpooling_layer_worker_specific_storage auws_;

  void aveunpool_set_params(size_t in_width,
                            size_t in_height,
                            size_t in_channels,
                            size_t pooling_size,
                            size_t stride) {
    params_.in  = shape3d(in_width, in_height, in_channels);
    params_.out = shape3d((in_width - 1) * stride + pooling_size,
                          (in_height - 1) * stride + pooling_size, in_channels);
    params_.window       = shape3d(pooling_size, pooling_size, in_channels);
    params_.stride       = stride;
    params_.scale_factor = sqr(pooling_size);
  }

  void init_connection(size_t pooling_size) {
    for (size_t c = 0; c < params_.in.depth_; ++c) {
      for (size_t y = 0; y < params_.in.height_; ++y) {
        for (size_t x = 0; x < params_.in.width_; ++x) {
          connect_kernel(pooling_size, x, y, c);
        }
      }
    }

    for (size_t c = 0; c < params_.in.depth_; ++c) {
      for (size_t y = 0; y < params_.out.height_; ++y) {
        for (size_t x = 0; x < params_.out.width_; ++x) {
          auws_.bias2out[c].push_back(params_.out.get_index(x, y, c));
        }
      }
    }
  }

  void connect_kernel(size_t pooling_size, size_t x, size_t y, size_t inc) {
    size_t dymax = std::min(pooling_size, params_.out.height_ - y);
    size_t dxmax = std::min(pooling_size, params_.out.width_ - x);
    size_t dstx  = x * params_.stride;
    size_t dsty  = y * params_.stride;
    size_t inidx = params_.in.get_index(x, y, inc);
    for (size_t dy = 0; dy < dymax; ++dy) {
      for (size_t dx = 0; dx < dxmax; ++dx) {
        connect_weight(inidx, params_.out.get_index(dstx + dx, dsty + dy, inc),
                       inc);
      }
    }
  }

  void connect_weight(size_t input_index,
                      size_t output_index,
                      size_t weight_index) {
    auws_.weight2io[weight_index].emplace_back(input_index, output_index);
    auws_.out2wi[output_index].emplace_back(weight_index, input_index);
    auws_.in2wo[input_index].emplace_back(weight_index, output_index);
  }
};

}  // namespace tiny_dnn
