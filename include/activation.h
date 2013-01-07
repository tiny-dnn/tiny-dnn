#pragma once
#include "util.h"

namespace tiny_cnn {

class activation {
public:
    virtual float_t f(float_t x) const = 0;
    virtual float_t df(float_t f_x) const = 0;
    virtual std::pair<float_t, float_t> scale() const = 0;
};

class identity_activation : public activation {
public:
    float_t f(float_t x) const { return x; }
    float_t df(float_t f_x) const { return 1; }  
    std::pair<float_t, float_t> scale() const { return std::make_pair(0.1, 0.9); }
};

class sigmoid_activation : public activation {
public:
    float_t f(float_t x) const { return 1.0 / (1.0 + std::exp(-x)); }
    float_t df(float_t f_x) const { return f_x * (1.0 - f_x); }
    std::pair<float_t, float_t> scale() const { return std::make_pair(0.1, 0.9); }
};

class tanh_activation : public activation {
public:
    float_t f(float_t x) const {
        const float_t ep = std::exp(x);
        const float_t em = std::exp(-x); 
        return (ep - em) / (ep + em);
    }

    // fast approximation of tanh (improve 2-3% speed in LeNet-5)
    /*float_t f(float_t x) const {
        const float_t x2 = x * x;
        x *= 1.0 + x2 * (0.1653 + x2 * 0.0097);
        return x / std::sqrt(1.0 + x * x);// invsqrt(static_cast<float>(1.0 + x * x));
    }*/

    float_t df(float_t f_x) const { return 1.0 - f_x * f_x; }
    std::pair<float_t, float_t> scale() const { return std::make_pair(-0.8, 0.8); }

private:
    /*float invsqrt(float x) const {
        float x2 = x * 0.5f;
        long i = *reinterpret_cast<long*>(&x);

        i = 0x5f3759df - (i >> 1);
        x = *reinterpret_cast<float*>(&i);
        x = x * (1.5f - (x2 * x * x));
        return x;
    }*/
};

class cost_function {
public:
    virtual float_t f(float_t y, float_t t) = 0;
    virtual float_t df(float_t y, float_t t) = 0;
};

class mse {
public:
    float_t f(float_t y, float_t t) {
        return (y - t) * (y - t) / 2;
    }

    float_t df(float_t y, float_t t) {
        return y - t;
    }
};

class cross_entropy {
public:
    float_t f(float_t y, float_t t) {
        throw nn_error("not implemented");
    }

    float_t df(float_t y, float_t t) {
        return (y - t) / (y * (1 - y));
    }
};

}
