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
#include <algorithm>

extern bool g_log_softmax;


namespace tiny_cnn {

/**
 * f(x) = h(scale*x+bias)
 */
template<typename Activation>
class linear_layer : public layer<Activation> {
public:
    CNN_USE_LAYER_MEMBERS;

    typedef layer<Activation> Base;

    explicit linear_layer(cnn_size_t dim, float_t scale = float_t(1), float_t bias = float_t(0))
        : Base(dim, dim, 0, 0),
        scale_(scale), bias_(bias) {}

    size_t param_size() const override {
        return 0;
    }

    size_t connection_size() const override {
        return this->in_size();
    }

    size_t fan_in_size() const override {
        return 1;
    }

    size_t fan_out_size() const override {
        return 1;
    }

    std::string layer_type() const override { return "linear"; }

    const vec_t& forward_propagation(const vec_t& in, size_t index) override {
        vec_t& a = a_[index];
        vec_t& out = output_[index];

        for_i(parallelize_, out_size_, [&](int i) {
            a[i] = scale_ * in[i] + bias_;
        });
        for_i(parallelize_, out_size_, [&](int i) {
            out[i] = h_.f(a, i);
        });

        return next_ ? next_->forward_propagation(out, index) : out;
    }

    virtual const vec_t& back_propagation(const vec_t& current_delta, size_t index) override {
        const vec_t& prev_out = prev_->output(index);
        const activation::function& prev_h = prev_->activation_function();
        vec_t& prev_delta = prev_delta_[index];

        for_i(parallelize_, out_size_, [&](int i) {
            prev_delta[i] = current_delta[i] * scale_ * prev_h.df(prev_out[i]);
        });

        return prev_->back_propagation(prev_delta_[index], index);
    }

    const vec_t& back_propagation_2nd(const vec_t& current_delta2) override {
        const vec_t& prev_out = prev_->output(0);
        const activation::function& prev_h = prev_->activation_function();

        for_i(parallelize_, out_size_, [&](int i) {
            prev_delta2_[i] = current_delta2[i] * sqr(scale_ * prev_h.df(prev_out[i]));
        });

        return prev_->back_propagation_2nd(prev_delta2_);
    }

protected:
    float_t scale_, bias_;
};

} // namespace tiny_cnn
