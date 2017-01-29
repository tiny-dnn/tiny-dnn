#pragma once
#include "tiny_dnn/activations/activation_layer.h"
#include "tiny_dnn/layers/layer.h"

namespace tiny_dnn {

class relu_layer : public activation_layer {
 public:
  using activation_layer::activation_layer;

  std::string layer_type() const override { return "relu-activation"; }

  void forward_activation(const vec_t &x, vec_t &y) {
    for (serial_size_t j = 0; j < x.size(); j++) {
      y[j] = std::max(float_t(0), x[j]);
    }
  }

  void backward_activation(const vec_t x,
                           const vec_t &y,
                           vec_t &dx,
                           const vec_t &dy) {
    for (serial_size_t j = 0; j < x.size(); j++) {
      float_t relu_grad = y[j] > float_t(0) ? float_t(1) : float_t(0);
      dx[j]             = dy[j] * relu_grad;
    }
  }
};
}  // namespace tiny_dnn
