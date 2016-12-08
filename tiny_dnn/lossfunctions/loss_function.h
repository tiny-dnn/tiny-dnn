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
#include "tiny_dnn/util/util.h"

namespace tiny_dnn {

// mean-squared-error loss function for regression
class mse {
public:
    static float_t f(const vec_t& y, const vec_t& t) {
        assert(y.size() == t.size());
        float_t d = 0.0;

        for(serial_size_t i = 0; i < y.size(); ++i)
            d += (y[i] - t[i]) * (y[i] - t[i]);

        return d/y.size();
    }

    static vec_t df(const vec_t& y, const vec_t& t) {
        assert(y.size() == t.size());
        vec_t d(t.size());
        float_t factor = float_t(2) / static_cast<float_t>(t.size());

        for(serial_size_t i = 0; i < y.size(); ++i)
            d[i] = factor * (y[i] - t[i]);

        return d;
    }
};

// absolute loss function for regression
class absolute {
public:
    static float_t f(const vec_t& y, const vec_t& t) {
        assert(y.size() == t.size());
        float_t d = float_t(0);

        for(serial_size_t i = 0; i < y.size(); ++i)
            d += std::abs(y[i] - t[i]);

        return d/y.size();
    }

    static vec_t df(const vec_t& y, const vec_t& t) {
        assert(y.size() == t.size());
        vec_t d(t.size());
        float_t factor = float_t(1) / static_cast<float_t>(t.size());

        for(serial_size_t i = 0; i < y.size(); ++i) {
            float_t sign = y[i] - t[i];
            if(sign < 0.f)
                d[i] = -float_t(1) * factor;
            else if(sign > 0.f)
                d[i] =  float_t(1) * factor;
            else
                d[i] =  float_t(0);
        }

        return d;
    }
};

// absolute loss with epsilon range for regression
// epsilon range [-eps, eps] with eps = 1./fraction
template<int fraction>
class absolute_eps {
public:
    static float_t f(const vec_t& y, const vec_t& t) {
        assert(y.size() == t.size());
        float_t d = float_t(0);
        const float_t eps = float_t(1) / fraction;

        for(serial_size_t i = 0; i < y.size(); ++i) {
            float_t diff = std::abs(y[i] - t[i]);
            if(diff > eps)
                d += diff;
        }
        return d / y.size();
    }

    static vec_t df(const vec_t& y, const vec_t& t) {
        assert(y.size() == t.size());
        vec_t d(t.size());
        const float_t factor = float_t(1) / static_cast<float_t>(t.size());
        const float_t eps    = float_t(1) / fraction;

        for(serial_size_t i = 0; i < y.size(); ++i) {
            float_t sign = y[i] - t[i];
            if(sign < -eps)
                d[i] = -float_t(1) * factor;
            else if(sign > eps)
                d[i] =  float_t(1) * factor;
            else
                d[i] = 0.f;
        }
        return d;
    }
};

// cross-entropy loss function for (multiple independent) binary classifications
class cross_entropy {
public:
    static float_t f(const vec_t& y, const vec_t& t) {
        assert(y.size() == t.size());
        float_t d = float_t(0);

        for(serial_size_t i = 0; i < y.size(); ++i)
            d += -t[i] * std::log(y[i]) - (float_t(1) - t[i]) * std::log(float_t(1) - y[i]);

        return d;
    }

    static vec_t df(const vec_t& y, const vec_t& t) {
        assert(y.size() == t.size());
        vec_t d(t.size());

        for(serial_size_t i = 0; i < y.size(); ++i)
            d[i] = (y[i] - t[i]) / (y[i] * (float_t(1) - y[i]));

        return d;
    }
};

// cross-entropy loss function for multi-class classification
class cross_entropy_multiclass {
public:
    static float_t f(const vec_t& y, const vec_t& t) {
        assert(y.size() == t.size());
        float_t d = 0.0;

        for(serial_size_t i = 0; i < y.size(); ++i)
            d += -t[i] * std::log(y[i]);

        return d;
    }

    static vec_t df(const vec_t& y, const vec_t& t) {
        assert(y.size() == t.size());
        vec_t d(t.size());

        for(serial_size_t i = 0; i < y.size(); ++i)
            d[i] = - t[i] / y[i];

        return d;
    }
};

template <typename E>
vec_t gradient(const vec_t& y, const vec_t& t) {
    assert(y.size() == t.size());
    return E::df(y, t);
}

template <typename E>
std::vector<vec_t> gradient(const std::vector<vec_t>& y, const std::vector<vec_t>& t) {
    std::vector<vec_t> grads;
 
    assert(y.size() == t.size());

    for (serial_size_t i = 0; i < y.size(); i++)
        grads.push_back(gradient<E>(y[i], t[i]));

    return grads;
}

inline void apply_cost_if_defined(std::vector<vec_t>& sample_gradient,
                                  const std::vector<vec_t>& sample_cost) {
    if (sample_gradient.size() == sample_cost.size()) {
        // @todo consider adding parallelism
        const serial_size_t channel_count = static_cast<serial_size_t>(sample_gradient.size());
        for (size_t channel = 0; channel < channel_count; ++channel) {
            if (sample_gradient[channel].size() == sample_cost[channel].size()) {
                const size_t element_count = sample_gradient[channel].size();

                // @todo optimize? (use AVX or so)
                for (size_t element = 0; element < element_count; ++element) {
                    sample_gradient[channel][element] *= sample_cost[channel][element];
                }
            }
        }
    }
}

// gradient for a minibatch
template <typename E>
std::vector<tensor_t> gradient(const std::vector<tensor_t>& y,
                               const std::vector<tensor_t>& t,
                               const std::vector<tensor_t>& t_cost) {

    const serial_size_t sample_count  = static_cast<serial_size_t>(y.size());
    const serial_size_t channel_count = static_cast<serial_size_t>(y[0].size());

    std::vector<tensor_t> gradients(sample_count);
 
    CNN_UNREFERENCED_PARAMETER(channel_count);
    assert(y.size() == t.size());
    assert(t_cost.empty() || t_cost.size() == t.size());

    // @todo add parallelism
    for (serial_size_t sample = 0; sample < sample_count; ++sample) {
        assert(y[sample].size() == channel_count);
        assert(t[sample].size() == channel_count);
        assert(t_cost.empty() || t_cost[sample].empty() ||
               t_cost[sample].size() == channel_count);

        gradients[sample] = gradient<E>(y[sample], t[sample]);

        if (sample < t_cost.size()) {
            apply_cost_if_defined(gradients[sample], t_cost[sample]);
        }
    }

    return gradients;
}

} // namespace tiny_dnn
