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

class leaky_relu_layer : public activation_layer {
 public:
  using activation_layer::activation_layer;

  std::string layer_type() const override { return "leaky-relu-activation"; }

  void forward_activation(const vec_t &x, vec_t &y) {
    for (serial_size_t j = 0; j < x.size(); j++) {
      y[j] = x[j] > float_t(0) ? x[j] : float_t(0.01) * x[j];
    }
  }

  void backward_activation(const vec_t &x,
                           const vec_t &y,
                           vec_t &dx,
                           const vec_t &dy) {
    for (serial_size_t j = 0; j < x.size(); j++) {
      // dx = dy * (gradient of leaky relu)
      dx[j] = dy[j] * (y[j] > float_t(0) ? float_t(1) : float_t(0.01));
    }
  }

  std::pair<float_t, float_t> scale() const override {
    return std::make_pair(float_t(0.1), float_t(0.9));
  }
};
}  // namespace tiny_dnn
