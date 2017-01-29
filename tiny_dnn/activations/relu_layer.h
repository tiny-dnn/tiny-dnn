#pragma once
#include "tiny_dnn/layers/layer.h"
#include "tiny_dnn/activations/activation_layer.h"

namespace tiny_dnn {

class relu_layer : public activation_layer {
public:
    using activation_layer::activation_layer;

    std::string layer_type() const override {
        return "relu-activation";
    }

    void forward_activation(const float_t &x, float_t &y) {
        y = std::max(float_t(0), x);
    }

    void backward_activation(const float_t x, const float_t &y,
                             float_t &dx, const float_t &dy) {
        float_t relu_grad = y > float_t(0) ? float_t(1) : float_t(0);
        dx = dy * relu_grad;
    }
};
}  // namespace tiny_dnn
