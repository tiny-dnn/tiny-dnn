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

class tanh_layer : public activation_layer {
 public:
  using activation_layer::activation_layer;

  std::string layer_type() const override { return "tanh-activation"; }

  void forward_activation(const vec_t &x, vec_t &y) {
    for (serial_size_t j = 0; j < x.size(); j++) {
      y[j] = std::tanh(x[j]);
    }
  }

  void backward_activation(const vec_t &x,
                           const vec_t &y,
                           vec_t &dx,
                           const vec_t &dy) {
    for (serial_size_t j = 0; j < x.size(); j++) {
      // dx = dy * (gradient of tanh)
      dx[j] = dy[j] * (float_t(1) - sqr(y[j]));
    }
  }
};
}  // namespace tiny_dnn
