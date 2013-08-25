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
#include <algorithm>

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

class rectified_linear : public activation {
public:
    float_t f(float_t x) const { return std::max((float_t)0.0, x); }
    float_t df(float_t f_x) const { return f_x > 0.0 ? 1.0 : 0.0; }
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

class softmax_activation : public activation {
public:

private:
};

} // namespace tiny_cnn
