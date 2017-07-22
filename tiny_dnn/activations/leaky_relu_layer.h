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

class leaky_relu_layer : public activation_layer {
 public:
  // using activation_layer::activation_layer;
  /**
   * Construct a leaky ReLU which will take shape when connected to some
   * layer. Connection happens like ( layer1 << act_layer1 ) and shape of this
   * layer is inferred at that time.
   */
  explicit leaky_relu_layer(const float_t epsilon = 0.01)
    : leaky_relu_layer(shape3d(0, 0, 0), epsilon) {}

  /**
   * Construct a flat leaky ReLU with specified number of neurons.
   * This constructor is suitable for adding an activation layer after
   * flat layers such as fully connected layers.
   *
   * @param in_dim      [in] number of elements of the input
   */
  explicit leaky_relu_layer(size_t in_dim, const float_t epsilon = 0.01)
    : leaky_relu_layer(shape3d(in_dim, 1, 1), epsilon) {}

  /**
   * Construct a leaky ReLU with specified width, height and channels.
   * This constructor is suitable for adding an activation layer after spatial
   * layers such as convolution / pooling layers.
   *
   * @param in_width    [in] number of input elements along width
   * @param in_height   [in] number of input elements along height
   * @param in_channels [in] number of channels (input elements along depth)
   */
  leaky_relu_layer(size_t in_width,
                   size_t in_height,
                   size_t in_channels,
                   const float_t epsilon = 0.01)
    : leaky_relu_layer(shape3d(in_width, in_height, in_channels), epsilon) {}

  /**
   * Construct a leaky ReLU layer with specified input shape.
   *
   * @param in_shape [in] shape of input tensor
   */
  explicit leaky_relu_layer(const shape3d &in_shape,
                            const float_t epsilon = 0.01)
    : activation_layer(in_shape), epsilon_(epsilon) {}

  /**
   * Construct a leaky ReLU layer given the previous layer.
   * @param prev_layer previous layer
   */
  explicit leaky_relu_layer(const layer &prev_layer,
                            const float_t epsilon = 0.01)
    : activation_layer(prev_layer), epsilon_(epsilon) {}

  std::string layer_type() const override { return "leaky-relu-activation"; }

  float_t epsilon_value() const { return epsilon_; }

  void forward_activation(const vec_t &x, vec_t &y) override {
    for (size_t j = 0; j < x.size(); j++) {
      y[j] = x[j] > float_t(0) ? x[j] : epsilon_ * x[j];
    }
  }

  void backward_activation(const vec_t &x,
                           const vec_t &y,
                           vec_t &dx,
                           const vec_t &dy) override {
    for (size_t j = 0; j < x.size(); j++) {
      // dx = dy * (gradient of leaky relu)
      dx[j] = dy[j] * (y[j] > float_t(0) ? float_t(1) : epsilon_);
    }
  }

  std::pair<float_t, float_t> scale() const override {
    return std::make_pair(float_t(0.1), float_t(0.9));
  }

  float_t epsilon_;
  friend struct serialization_buddy;
};

}  // namespace tiny_dnn
