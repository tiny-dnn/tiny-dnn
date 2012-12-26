#pragma once
#include "util.h"

namespace nn {

struct updater {
public:
    virtual void update(float_t dW, float_t H, float_t *W) = 0;
};

struct gradient_descent : public updater {
public:
    gradient_descent() : alpha(0.000085), lambda(0.0), mu(0.02) {}
    gradient_descent(double alpha, double lambda, double mu) : alpha(alpha), lambda(lambda), mu(mu) {}

    void update(float_t dW, float_t H, float_t *W) {
        *W = *W - (alpha / (H + mu)) * (dW + lambda);
    }

    double alpha; // learning rate
    double lambda; // weight decay
    double mu;
};

}
