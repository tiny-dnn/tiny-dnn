#pragma once

namespace nn {

struct sigmoid_activation {
    static double f(double x) { return 1.0 / (1.0 + std::exp(-x)); }
    static double df(double f_x) { return f_x * (1.0 - f_x); }
};

struct tanh_activation {
    static double f(double x) {
        const double ep = std::exp(x);
        const double em = std::exp(-x); 
        return (ep - em) / (ep + em);
    }
    static double df(double f_x) { return f_x * (1.0 - f_x); }
};

}
