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

class sigmoid_layer : public activation_layer {
 public:
  using activation_layer::activation_layer;

  std::string layer_type() const override { return "sigmoid-activation"; }

  void forward_activation(const ConstViewTensor x, ViewTensor y) override {
    auto itx = x.host_begin();
    auto ity = y.host_begin();
    for (; itx != x.host_end(); ++itx, ++ity) {
      *ity = float_t(1) / (float_t(1) + std::exp(-*itx));
    }
  }

  void backward_activation(const ConstViewTensor x,
                           const ConstViewTensor y,
                           ViewTensor dx,
                           const ConstViewTensor dy) override {
    auto itx  = x.host_begin();
    auto ity  = y.host_begin();
    auto itdx = dx.host_begin();
    auto itdy = dy.host_begin();
    for (; itdx != dx.host_end(); ++itx, ++ity, ++itdx, ++itdy) {
      // dx = dy * (gradient of sigmoid)
      *itdx = *itdy * *ity * (float_t(1) - *ity);
    }
  }

  std::pair<float_t, float_t> scale() const override {
    return std::make_pair(float_t(0.1), float_t(0.9));
  }

  friend struct serialization_buddy;
};

}  // namespace tiny_dnn
