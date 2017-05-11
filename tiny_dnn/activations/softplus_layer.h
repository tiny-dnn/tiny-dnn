/*
    Copyright (c) 2017, Taiga Nomi
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once
#include "tiny_dnn/activations/activation_layer.h"
#include "tiny_dnn/layers/layer.h"

namespace tiny_dnn {

class softplus_layer : public activation_layer {
 public:
  using activation_layer::activation_layer;

  std::string layer_type() const override { return "softplus-activation"; }

  void forward_activation(const vec_t &x, vec_t &y) override {
    for (serial_size_t j = 0; j < x.size(); j++) {
      // This is an approximation for numerical stability
      // log1p(exp(30)) = 30.000000000000092 and log1p(exp(-30)) = 9.3576229688397368e-14
      if (x[j] >= float_t(30)) {
        y[j] = x[j];
      } else if (x[j] <= float_t(-30)) {
        y[j] = float_t(0);
      } else {
        y[j] = std::log1p(std::exp(x[j]));
      }
    }
  }

  void backward_activation(const vec_t &x,
                           const vec_t &y,
                           vec_t &dx,
                           const vec_t &dy) override {
    for (serial_size_t j = 0; j < x.size(); j++) {
      // dx = dy * (gradient of softplus)
      dx[j] = dy[j] / (float_t(1) + std::exp(-x[j]));;
    }
  }

  std::pair<float_t, float_t> scale() const override {
    return std::make_pair(float_t(0.1), float_t(0.9));
  }

  friend struct serialization_buddy;
};

}  // namespace tiny_dnn
