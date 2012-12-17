#pragma once

namespace nn {

class learner {
public:
    virtual void update(pvec_t& target, const pvec_t& diff) = 0;
};

class gradient_descent : public learner {
public:
    gradient_descent(double alpha) : alpha_(alpha){}

    void update(pvec_t& target, const pvec_t& diff) {
        for (size_t i = 0; i < diff.size(); i++)
            *target[i] = *target[i] - alpha_ * *diff[i];
    }

private:
    double alpha_; // learning rate
};

}
