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

#if HAS_CXX11_THREAD_LOCAL
#define TINY_DNN_THREAD_LOCAL thread_local
#else
#define TINY_DNN_THREAD_LOCAL
#endif

namespace tiny_dnn {

class softmax_layer : public activation_layer {
 public:
  using activation_layer::activation_layer;

  std::string layer_type() const override { return "softmax-activation"; }

  void forward_activation(const ConstViewTensor x, ViewTensor y) override {
    auto itx            = x.host_begin();
    auto ity            = y.host_begin();
    const float_t alpha = *std::max_element(x.host_begin(), x.host_end());
    float_t denominator(0);
    for (; itx != x.host_end(); ++itx, ++ity) {
      *ity = std::exp(*itx - alpha);
      denominator += *ity;
    }
    for (ity = y.host_begin(); ity != y.host_end(); ++ity) {
      *ity /= denominator;
    }
  }

  void backward_activation(const ConstViewTensor x,
                           const ConstViewTensor y,
                           ViewTensor dx,
                           const ConstViewTensor dy) override {
    const size_t len = dy.shape()[0];

    // auxilliary vector to store element wise softmax gradients of all elements
    TINY_DNN_THREAD_LOCAL Tensor<> df({len});

    auto ity  = y.host_begin();
    auto itdx = dx.host_begin();
    for (; itdx != dx.host_end(); ++ity, ++itdx) {
      auto itdf = df.host_begin();
      auto ity2 = y.host_begin();
      for (; ity2 != y.host_end(); ++ity2, ++itdf) {
        *itdf = (ity == ity2) ? *ity * (float_t(1) - *ity) : -*ity2 * *ity;
      }
      // dx = dy * (gradient of softmax)
      *itdx = vectorize::dot(dy.host_begin(), df.host_begin(), len);
    }
  }

  std::pair<float_t, float_t> scale() const override {
    return std::make_pair(float_t(0), float_t(1));
  }

  friend struct serialization_buddy;
};

}  // namespace tiny_dnn
