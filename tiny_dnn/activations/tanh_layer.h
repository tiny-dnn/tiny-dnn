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

class tanh_layer : public activation_layer {
 public:
  using activation_layer::activation_layer;

  std::string layer_type() const override { return "tanh-activation"; }

  void forward_activation(const ConstViewTensor x, ViewTensor y) override {
    auto iterx = x.host_begin();
    auto itery = y.host_begin();
    for (; iterx != x.host_end(); ++iterx, ++itery) {
      *itery = *iterx;
    }
  }

  void backward_activation(const ConstViewTensor x,
                           const ConstViewTensor y,
                           ViewTensor dx,
                           const ConstViewTensor dy) override {
    auto iterdx = dx.host_begin();
    auto iterdy = dy.host_begin();
    auto itery  = y.host_begin();
    for (; iterdx != dx.host_end(); ++iterdx, ++iterdy, ++itery) {
      // dx = dy * (gradient of tanh)
      *iterdx = *iterdy * (float_t(1) - sqr(*itery));
    }
  }

  std::pair<float_t, float_t> scale() const override {
    return std::make_pair(float_t(-0.8), float_t(0.8));
  }

  friend struct serialization_buddy;
};

}  // namespace tiny_dnn
