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
class linear_layer : public feedforward_layer<Activation> {
public:
    CNN_USE_LAYER_MEMBERS;

    typedef feedforward_layer<Activation> Base;

    explicit linear_layer(cnn_size_t dim, float_t scale = float_t(1), float_t bias = float_t(0))
        : Base({vector_type::data}),
        dim_(dim), scale_(scale), bias_(bias) {}

    std::vector<shape3d> in_shape() const override {
        return {shape3d(dim_, 1, 1) };
    }

    std::vector<shape3d> out_shape() const override {
        return{ shape3d(dim_, 1, 1), shape3d(dim_, 1, 1) };
    }

    std::string layer_type() const override { return "linear"; }

    void forward_propagation(cnn_size_t index,
                             const std::vector<vec_t*>& in_data,
                             std::vector<vec_t*>& out_data) override {
        const vec_t& in  = *in_data[0];
        vec_t&       out = *out_data[0];
        vec_t&       a   = *out_data[1];

        CNN_UNREFERENCED_PARAMETER(index);

        for_i(parallelize_, dim_, [&](int i) {
            a[i] = scale_ * in[i] + bias_;
        });
        for_i(parallelize_, dim_, [&](int i) {
            out[i] = h_.f(a, i);
        });
    }

    void back_propagation(cnn_size_t                index,
                          const std::vector<vec_t*>& in_data,
                          const std::vector<vec_t*>& out_data,
                          std::vector<vec_t*>&       out_grad,
                          std::vector<vec_t*>&       in_grad) override {
        vec_t&       prev_delta = *in_grad[0];
        vec_t&       curr_delta = *out_grad[1];

        CNN_UNREFERENCED_PARAMETER(index);
        CNN_UNREFERENCED_PARAMETER(in_data);

        this->backward_activation(*out_grad[0], *out_data[0], curr_delta);

        for_i(parallelize_, dim_, [&](int i) {
            prev_delta[i] = curr_delta[i] * scale_;
        });
    }

   /* const vec_t& back_propagation_2nd(const vec_t& current_delta2) override {
        const vec_t& prev_out = prev_->output(0);
        const activation::function& prev_h = prev_->activation_function();

        for_i(parallelize_, out_size_, [&](int i) {
            prev_delta2_[i] = current_delta2[i] * sqr(scale_ * prev_h.df(prev_out[i]));
        });

        return prev_->back_propagation_2nd(prev_delta2_);
    }*/

protected:
    cnn_size_t dim_;
    float_t scale_, bias_;
};

} // namespace tiny_cnn
