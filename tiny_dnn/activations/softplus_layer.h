/*
    Copyright (c) 2017, Taiga Nomi
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

class softplus_layer : public activation_layer {
 public:
  // using activation_layer::activation_layer;
  /**
   * Construct a softplus which will take shape when connected to some
   * layer. Connection happens like ( layer1 << act_layer1 ) and shape of this
   * layer is inferred at that time.
   */
  explicit softplus_layer(const float_t beta      = 1.0,
                          const float_t threshold = 20.0)
    : softplus_layer(shape3d(0, 0, 0), beta, threshold) {}

  /**
   * Construct a flat softplus with specified number of neurons.
   * This constructor is suitable for adding an activation layer after
   * flat layers such as fully connected layers.
   *
   * @param in_dim      [in] number of elements of the input
   */
  softplus_layer(size_t in_dim,
                 const float_t beta      = 1.0,
                 const float_t threshold = 20.0)
    : softplus_layer(shape3d(in_dim, 1, 1), beta, threshold) {}

  /**
   * Construct a softplus with specified width, height and channels.
   * This constructor is suitable for adding an activation layer after spatial
   * layers such as convolution / pooling layers.
   *
   * @param in_width    [in] number of input elements along width
   * @param in_height   [in] number of input elements along height
   * @param in_channels [in] number of channels (input elements along depth)
   */
  softplus_layer(size_t in_width,
                 size_t in_height,
                 size_t in_channels,
                 const float_t beta      = 1.0,
                 const float_t threshold = 20.0)
    : softplus_layer(
        shape3d(in_width, in_height, in_channels), beta, threshold) {}

  /**
   * Construct a softplus layer with specified input shape.
   *
   * @param in_shape [in] shape of input tensor
   */
  softplus_layer(const shape3d &in_shape,
                 const float_t beta      = 1.0,
                 const float_t threshold = 20.0)
    : activation_layer(in_shape), beta_(beta), threshold_(threshold) {}

  /**
   * Construct a softplus layer given the previous layer.
   * @param prev_layer previous layer
   */
  softplus_layer(const layer &prev_layer,
                 const float_t beta      = 1.0,
                 const float_t threshold = 20.0)
    : activation_layer(prev_layer), beta_(beta), threshold_(threshold) {}

  std::string layer_type() const override { return "softplus-activation"; }

  float_t beta_value() const { return beta_; }

  float_t threshold_value() const { return threshold_; }

  void forward_activation(const vec_t &x, vec_t &y) override {
    for (size_t j = 0; j < x.size(); j++) {
      float_t betain = beta_ * x[j];
      y[j]           = (betain > threshold_) ? x[j]
                                   : (1 / beta_) * std::log1p(std::exp(betain));
    }
  }

  void backward_activation(const vec_t &x,
                           const vec_t &y,
                           vec_t &dx,
                           const vec_t &dy) override {
    for (size_t j = 0; j < x.size(); j++) {
      float_t betaout = beta_ * y[j];
      float_t exp_bo  = std::exp(betaout);
      // dx = dy * (gradient of softplus)
      dx[j] = (betaout > threshold_) ? dy[j] : dy[j] * (exp_bo - 1) / exp_bo;
    }
  }

  std::pair<float_t, float_t> scale() const override {
    return std::make_pair(float_t(0.1), float_t(0.9));
  }

  float_t beta_;
  float_t threshold_;
  friend struct serialization_buddy;
};

}  // namespace tiny_dnn
