#pragma once
#include "util.h"

namespace nn {

class learner {
public:
    virtual void update(float_t dW, float_t *W) = 0;
};

class gradient_descent : public learner {
public:
    gradient_descent(double alpha) : alpha_(alpha){}

    void update(float_t dW, float_t *W) {
        *W = *W - alpha_ * dW;
    }

private:
    double alpha_; // learning rate
};

}
