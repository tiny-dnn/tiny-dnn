#pragma once
#include "util.h"

namespace nn {

class activation {
public:
    virtual double f(double x) const = 0;
    virtual double df(double f_x) const = 0;
    virtual std::pair<double, double> scale() const = 0;
};

class identity_activation : public activation {
public:
    double f(double x) const { return x; }
    double df(double f_x) const { return 1; }  
    std::pair<double, double> scale() const { return std::make_pair(0.2, 0.8); }
};

class sigmoid_activation : public activation {
public:
    double f(double x) const { return 1.0 / (1.0 + std::exp(-x)); }
    double df(double f_x) const { return f_x * (1.0 - f_x); }
    std::pair<double, double> scale() const { return std::make_pair(0.2, 0.8); }
};

class tanh_activation : public activation {
public:
    double f(double x) const {
        const double ep = std::exp(x);
        const double em = std::exp(-x); 
        return (ep - em) / (ep + em);
    }
    double df(double f_x) const { return f_x * (1.0 - f_x); }
    std::pair<double, double> scale() const { return std::make_pair(-0.8, 0.8); }
};

class cost_function {
public:
    virtual float_t f(float_t y, float_t t) = 0;
    virtual float_t df(float_t y, float_t t) = 0;
};

class mse : public cost_function {
public:
    float_t f(float_t y, float_t t) {
        return (y - t) * (y - t) / 2;
    }

    float_t df(float_t y, float_t t) {
        return y - t;
    }

private:
    double lambda_;
};

class cross_entropy : public cost_function {
public:
    float_t f(float_t y, float_t t) {
        return 0.0;//TODO
    }

    float_t df(float_t y, float_t t) {
        return (y - t) / (y * (1 - y));
    }
};

}
