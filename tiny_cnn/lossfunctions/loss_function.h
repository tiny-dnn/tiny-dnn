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
#include "tiny_cnn/util/util.h"

namespace tiny_cnn {

// mean-squared-error loss function for regression
class mse {
public:
    static float_t f(float_t y, float_t t) {
        return (y - t) * (y - t) / 2;
    }

    static float_t df(float_t y, float_t t) {
        return y - t;
    }
};

// cross-entropy loss function for (multiple independent) binary classifications
class cross_entropy {
public:
    static float_t f(float_t y, float_t t) {
        return -t * std::log(y) - (float_t(1) - t) * std::log(float_t(1) - y);
    }

    static float_t df(float_t y, float_t t) {
        return (y - t) / (y * (float_t(1) - y));
    }
};

// cross-entropy loss function for multi-class classification
class cross_entropy_multiclass {
public:
    static float_t f(float_t y, float_t t) {
        return -t * std::log(y);
    }

    static float_t df(float_t y, float_t t) {
        return - t / y;
    }
};

template <typename E>
vec_t gradient(const vec_t& y, const vec_t& t) {
    vec_t grad(y.size());
    assert(y.size() == t.size());

    for (cnn_size_t i = 0; i < y.size(); i++)
        grad[i] = E::df(y[i], t[i]);

    return grad;
}

} // namespace tiny_cnn
