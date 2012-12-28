#pragma once
#include "util.h"

namespace nn {

struct updater {
public:
    virtual void update(float_t dW, float_t H, float_t *W) = 0;
};

struct gradient_descent : public updater {
public:
    gradient_descent() : alpha(0.00085), lambda(0.0), mu(0.02) {}
    gradient_descent(float_t alpha, float_t lambda, float_t mu) : alpha(alpha), lambda(lambda), mu(mu) {}

    void update(float_t dW, float_t H, float_t *W) {
        *W -= (alpha / (H + mu)) * (dW + lambda); // 7.2%
    }

    const float_t alpha; // learning rate
    const float_t lambda; // weight decay
    const float_t mu;
};

}
