#pragma once
#include "util.h"

namespace nn {

class activation {
public:
    virtual double f(double x) const = 0;
    virtual double df(double f_x) const = 0;
};

class sigmoid_activation : public activation {
public:
    double f(double x) const { return 1.0 / (1.0 + std::exp(-x)); }
    double df(double f_x) const { return f_x * (1.0 - f_x); }
};

class tanh_activation : public activation {
public:
    double f(double x) const {
        const double ep = std::exp(x);
        const double em = std::exp(-x); 
        return (ep - em) / (ep + em);
    }
    double df(double f_x) const { return f_x * (1.0 - f_x); }
};

class cost_function {
public:
    virtual float_t f(const vec_t& y, const vec_t& t) = 0;
    virtual float_t df(float_t yi, float_t ti) = 0;
};

class mse : public cost_function{
    float_t f(const vec_t& y, const vec_t& t) {
        float_t e = 0.0;
        for (size_t i = 0; i < y.size(); i++) {
            e += (y[i] - t[i]) * (y[i] - t[i]);
        }
        e /= 2.0;
        return e;
    }

    float_t df(float_t yi, float_t ti) {
        return yi - ti;
    }

private:
    double lambda_;
};

}
