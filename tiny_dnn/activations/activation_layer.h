/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "tiny_dnn/layers/layer.h"
#include "tiny_dnn/util/util.h"

namespace tiny_dnn {

class activation_layer : public layer {
 public:
  /**
   * Construct an activation layer which will take shape when connected to some
   * layer. Connection happens like ( layer1 << act_layer1 ) and shape of this
   * layer is inferred at that time.
   */
  activation_layer() : activation_layer(shape3d(0, 0, 0)) {}

  /**
   * Construct a flat activation layer with specified number of neurons.
   * This constructor is suitable for adding an activation layer after
   * flat layers such as fully connected layers.
   *
   * @param in_dim      [in] number of elements of the input
   */
  explicit activation_layer(size_t in_dim)
    : activation_layer(shape3d(in_dim, 1, 1)) {}

  /**
   * Construct an activation layer with specified width, height and channels.
   * This constructor is suitable for adding an activation layer after spatial
   * layers such as convolution / pooling layers.
   *
   * @param in_width    [in] number of input elements along width
   * @param in_height   [in] number of input elements along height
   * @param in_channels [in] number of channels (input elements along depth)
   */
  activation_layer(size_t in_width, size_t in_height, size_t in_channels)
    : activation_layer(shape3d(in_width, in_height, in_channels)) {}

  /**
   * Construct an activation layer with specified input shape.
   *
   * @param in_shape [in] shape of input tensor
   */
  explicit activation_layer(const shape3d &in_shape)
    : layer({vector_type::data}, {vector_type::data}), in_shape_(in_shape) {}

  /**
   * Construct an activation layer given the previous layer.
   * @param prev_layer previous layer
   */
  explicit activation_layer(const layer &prev_layer)
    : layer({vector_type::data}, {vector_type::data}),
      in_shape_(prev_layer.out_shape()[0]) {}

  std::vector<shape3d> in_shape() const override { return {in_shape_}; }

  std::vector<shape3d> out_shape() const override { return {in_shape_}; }

  void set_in_shape(const shape3d &in_shape) override {
    this->in_shape_ = in_shape;
  }

  void forward_propagation(const std::vector<tensor_t *> &in_data,
                           std::vector<tensor_t *> &out_data) override {
    const tensor_t &x = *in_data[0];
    tensor_t &y       = *out_data[0];
    for_i(x.size(), [&](size_t i) { forward_activation(x[i], y[i]); });
  }

  void back_propagation(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) override {
    tensor_t &dx       = *in_grad[0];
    const tensor_t &dy = *out_grad[0];
    const tensor_t &x  = *in_data[0];
    const tensor_t &y  = *out_data[0];
    for_i(x.size(),
          [&](size_t i) { backward_activation(x[i], y[i], dx[i], dy[i]); });
  }

  std::string layer_type() const override = 0;

  /**
   * Populate vec_t of elements 'y' according to activation y = f(x).
   * Child classes must override this method, apply activation function
   * element wise over a vec_t of elements.
   *
   * @param x  input vector
   * @param y  output vector (values to be assigned based on input)
   **/
  virtual void forward_activation(const vec_t &x, vec_t &y) = 0;

  /**
   * Populate vec_t of elements 'dx' according to gradient of activation.
   *
   * @param x  input vector of current layer (same as forward_activation)
   * @param y  output vector of current layer (same as forward_activation)
   * @param dx gradient of input vectors (i-th element correspond with x[i])
   * @param dy gradient of output vectors (i-th element correspond with y[i])
   **/
  virtual void backward_activation(const vec_t &x,
                                   const vec_t &y,
                                   vec_t &dx,
                                   const vec_t &dy) = 0;

  /**
   * Target value range for learning.
   */
  virtual std::pair<float_t, float_t> scale() const = 0;

 private:
  shape3d in_shape_;
};

}  // namespace tiny_dnn
