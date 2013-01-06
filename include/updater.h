#pragma once
#include "util.h"

namespace tiny_cnn {

/*struct updater {
public:
    virtual void update(float_t dW, float_t H, float_t *W) = 0;
};*/

struct gradient_descent {
public:
    gradient_descent() : alpha(0.00085), mu(0.02) {}
    gradient_descent(float_t alpha, float_t lambda, float_t mu) : alpha(alpha), mu(mu) {}

    void update(float_t dW, float_t H, float_t *W) {
        *W -= (alpha / (H + mu)) * (dW); // 7.2%
    }

    float_t alpha; // learning rate
    //const float_t lambda; // weight decay
    float_t mu;
};

}
