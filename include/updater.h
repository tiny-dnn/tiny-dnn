#pragma once
#include "util.h"

namespace nn {

struct updater {
public:
    virtual void update(float_t dW, float_t *W) = 0;
};

struct gradient_descent : public updater {
public:
    gradient_descent() : alpha(0.1), lambda(0.0) {}
    gradient_descent(double alpha, double lambda) : alpha(alpha), lambda(lambda) {}

    void update(float_t dW, float_t *W) {
        *W = *W - alpha * (dW + lambda);
    }

    double alpha; // learning rate
    double lambda; // weight decay
};

}
