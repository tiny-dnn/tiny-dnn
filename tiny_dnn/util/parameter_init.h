/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/framework/tensor.h"
#include "tiny_dnn/util/util.h"

namespace tiny_dnn {
namespace parameter_init {

// todo (karandesai) : refer and exercise
// https://github.com/alykhantejani/nninit/blob/master/nninit.py

class function {
 public:
  virtual void fill(Tensor<> &tensor,
                    const size_t &fan_in,
                    const size_t &fan_out) {}
};

class scalable : public function {
 public:
  explicit scalable(float_t value) : scale_(value) {}

  void scale(float_t value) { scale_ = value; }

 protected:
  float_t scale_;
};

/**
 * Use fan-in and fan-out for scaling
 *
 * X Glorot, Y Bengio,
 * Understanding the difficulty of training deep feedforward neural networks
 * Proc. AISTATS 10, May 2010, vol.9, pp249-256
 **/
class xavier : public scalable {
 public:
  xavier() : scalable(float_t{6}) {}
  explicit xavier(float_t value) : scalable(value) {}

  void fill(Tensor<> &tensor,
            const size_t &fan_in,
            const size_t &fan_out) override {
    const float_t weight_base = std::sqrt(scale_ / (fan_in + fan_out));

    uniform_rand(tensor.host_begin(), tensor.host_end(), -weight_base,
                 weight_base);
  }
};

class constant : public scalable {
 public:
  constant() : scalable(float_t{0}) {}
  explicit constant(float_t value) : scalable(value) {}

  void fill(Tensor<> &tensor,
            const size_t &fan_in,
            const size_t &fan_out) override {
    CNN_UNREFERENCED_PARAMETER(fan_in);
    CNN_UNREFERENCED_PARAMETER(fan_out);
    tensor.fill(scale_);
  }
};

/**
 * Use fan-in(number of input weight for each neuron) for scaling
 *
 * Y LeCun, L Bottou, G B Orr, and K Muller,
 * Efficient backprop
 * Neural Networks, Tricks of the Trade, Springer, 1998
 **/
class lecun : public scalable {
 public:
  lecun() : scalable(float_t{1}) {}
  explicit lecun(float_t value) : scalable(value) {}

  void fill(Tensor<float_t> &tensor,
            const size_t &fan_in,
            const size_t &fan_out) override {
    CNN_UNREFERENCED_PARAMETER(fan_out);

    const float_t weight_base = scale_ / std::sqrt(float_t(fan_in));

    uniform_rand(tensor.host_begin(), tensor.host_end(), -weight_base,
                 weight_base);
  }
};

class gaussian : public scalable {
 public:
  gaussian() : scalable(float_t{1}) {}
  explicit gaussian(float_t sigma) : scalable(sigma) {}

  void fill(Tensor<float_t> &tensor,
            const size_t &fan_in,
            const size_t &fan_out) override {
    CNN_UNREFERENCED_PARAMETER(fan_in);
    CNN_UNREFERENCED_PARAMETER(fan_out);

    gaussian_rand(tensor.host_begin(), tensor.host_end(), float_t{0}, scale_);
  }
};

class he : public scalable {
 public:
  he() : scalable(float_t{2}) {}
  explicit he(float_t value) : scalable(value) {}

  void fill(Tensor<float_t> &tensor,
            const size_t &fan_in,
            const size_t &fan_out) override {
    CNN_UNREFERENCED_PARAMETER(fan_out);

    const float_t sigma = std::sqrt(scale_ / fan_in);
    gaussian_rand(tensor.host_begin(), tensor.host_end(), float_t{0}, sigma);
  }
};

}  // namespace parameter_init
}  // namespace tiny_dnn
