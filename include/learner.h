#pragma once
#include "util.h"

namespace nn {

struct learner {
public:
    virtual void update(float_t dW, float_t *W) = 0;
};

struct gradient_descent : public learner {
public:
    gradient_descent() : alpha(0.3){}
    gradient_descent(double alpha) : alpha(alpha){}

    void update(float_t dW, float_t *W) {
        *W = *W - alpha * dW;
    }

    double alpha; // learning rate
    double lambda; // weight decay
};

}
