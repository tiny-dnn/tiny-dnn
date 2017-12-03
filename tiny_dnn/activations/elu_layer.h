/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <string>
#include <utility>

#include "tiny_dnn/activations/activation_layer.h"
#include "tiny_dnn/layers/layer.h"

namespace tiny_dnn {

class elu_layer : public activation_layer {
 public:
  explicit elu_layer(const float_t alpha = 1.0)
    : elu_layer(shape3d(0, 0, 0), alpha) {}

  explicit elu_layer(size_t in_dim, const float_t alpha = 1.0)
    : elu_layer(shape3d(in_dim, 1, 1), alpha) {}

  elu_layer(size_t in_width,
            size_t in_height,
            size_t in_channels,
            const float_t alpha = 1.0)
    : elu_layer(shape3d(in_width, in_height, in_channels), alpha) {}

  explicit elu_layer(const shape3d &in_shape, const float_t alpha = 1.0)
    : activation_layer(in_shape), alpha_(alpha) {}

  explicit elu_layer(const layer &prev_layer, const float_t alpha = 1.0)
    : activation_layer(prev_layer), alpha_(alpha) {}

  std::string layer_type() const override { return "elu-activation"; }

  void forward_activation(const vec_t &x, vec_t &y) override {
    for (size_t j = 0; j < x.size(); j++) {
      y[j] =
        x[j] < float_t(0) ? (alpha_ * (std::exp(x[j]) - float_t(1))) : x[j];
    }
  }

  void backward_activation(const vec_t &x,
                           const vec_t &y,
                           vec_t &dx,
                           const vec_t &dy) override {
    for (size_t j = 0; j < x.size(); j++) {
      // dx = dy * (gradient of elu)
      dx[j] = dy[j] * (y[j] > float_t(0) ? float_t(1) : (alpha_ + y[j]));
    }
  }

  std::pair<float_t, float_t> scale() const override {
    return std::make_pair(float_t(0.1), float_t(0.9));
  }

  float_t alpha_;
  friend struct serialization_buddy;
};

}  // namespace tiny_dnn
