/*
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY 
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND 
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#include "util.h"
#include <unordered_map>

namespace tiny_cnn {

struct optimizer {
    virtual bool requires_hessian() const { return true; }
    virtual void reset() {}
};

// gradient descent with 2nd-order update(LeCun,1998)
struct gradient_descent_levenberg_marquardt : public optimizer {
public:
    gradient_descent_levenberg_marquardt() : alpha(0.00085), mu(0.02) {}
    gradient_descent_levenberg_marquardt(float_t alpha, float_t mu) : alpha(alpha), mu(mu) {}

    void update(const vec_t& dW, const vec_t& Hessian, vec_t *W) {
        for_(true, 0, W->size(), [&](const blocked_range& r){
            for (int i = r.begin(); i < r.end(); i++)
                update_(dW[i], Hessian[i], &(*W)[i]);
        });
    }

    float_t alpha; // learning rate
    float_t mu;

private:
    void update_(float_t dW, float_t H, float_t *W) {
        *W -= (alpha / (H + mu)) * (dW); // 7.2%
    }
};

struct adagrad : public optimizer {
    adagrad() : alpha(0.01) {}
    explicit adagrad(float_t alpha) : alpha(alpha) {}

    bool requires_hessian() const {
        return false;
    }

    void reset() {
        E_.clear();
    }

    void update(const vec_t& dW, const vec_t& /*Hessian*/, vec_t *W) {
        vec_t& E = E_[&dW];
        if (E.empty()) {
            E.resize(dW.size(), float_t());
        }

        for_(true, 0, W->size(), [&](const blocked_range& r){
            for (int i = r.begin(); i < r.end(); i++) {
                E[i] += dW[i] * dW[i];
                (*W)[i] -= alpha * dW[i] / std::sqrt(E[i]);
            }
        });
    }

    float_t alpha; // learning rate
private:
    std::unordered_map<const vec_t*, vec_t> E_; // sum of squares of gradients for each layer
};

// simple SGD algorithm
struct gradient_descent : public optimizer {
public:
    gradient_descent() : alpha(0.01), lambda(0.0) {}
    gradient_descent(float_t alpha, float_t lambda) : alpha(alpha), lambda(lambda) {}

    void update(const vec_t& dW, const vec_t& /*Hessian*/, vec_t *W) {
        for_(true, 0, W->size(), [&](const blocked_range& r){
            for (int i = r.begin(); i < r.end(); i++)
                update_(dW[i], &(*W)[i]);
        });
    }

    bool requires_hessian() const {
        return false;
    }

    float_t alpha; // learning rate
    float_t lambda; // weight decay

private:
    void update_(float_t dW, float_t *W) {
        *W -= alpha * ((dW) +*W * lambda); // 7.2%
    }
};

} // namespace tiny_cnn
