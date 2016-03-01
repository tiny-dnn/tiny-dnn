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

namespace tiny_cnn {

    enum class norm_region {
        across_channels,
        within_channels
    };

/**
 * local response normalization
 */
template<typename Activation>
class lrn_layer : public layer<Activation> {
public:
    CNN_USE_LAYER_MEMBERS;

    typedef layer<Activation> Base;

    lrn_layer(cnn_size_t in_width, cnn_size_t in_height, cnn_size_t local_size, cnn_size_t in_channels,
                       float_t alpha, float_t beta, norm_region region = norm_region::across_channels)
        : Base(in_width*in_height*in_channels, in_width*in_height*in_channels, 0, 0),
        in_shape_(in_width, in_height, in_channels), size_(local_size), alpha_(alpha), beta_(beta), region_(region), in_square_(in_shape_.area()) {}

    size_t param_size() const override {
        return 0;
    }

    size_t connection_size() const override {
        return this->in_size() * size_;
    }

    size_t fan_in_size() const override {
        return size_;
    }

    size_t fan_out_size() const override {
        return size_;
    }

    std::string layer_type() const override { return "norm"; }

    const vec_t& forward_propagation(const vec_t& in, size_t index) override {
        vec_t& a = a_[index];
        vec_t& out = output_[index];

        if (region_ == norm_region::across_channels) {
            forward_across(in, a);
        }
        else {
            forward_within(in, a);
        }

        for_i(parallelize_, out_size_, [&](int i) {
            out[i] = h_.f(a, i);
        });
        return next_ ? next_->forward_propagation(out, index) : out;
    }

    virtual const vec_t& back_propagation(const vec_t& current_delta, size_t index) override {
        CNN_UNREFERENCED_PARAMETER(current_delta);
        CNN_UNREFERENCED_PARAMETER(index);
        throw nn_error("not implemented");
    }

    const vec_t& back_propagation_2nd(const vec_t& current_delta2) override {
        CNN_UNREFERENCED_PARAMETER(current_delta2);
        throw nn_error("not implemented");
    }

private:
    void forward_across(const vec_t& in, vec_t& out) {
        std::fill(in_square_.begin(), in_square_.end(), float_t(0));

        for (cnn_size_t i = 0; i < size_ / 2; i++) {
            cnn_size_t idx = in_shape_.get_index(0, 0, i);
            add_square_sum(&in[idx], in_shape_.area(), &in_square_[0]);
        }

        cnn_size_t head = size_ / 2;
        long tail = ((long) head) - size_;
        cnn_size_t channels = in_shape_.depth_;
        const cnn_size_t wxh = in_shape_.area();
        const float_t alpha_div_size = alpha_ / size_;

        for (cnn_size_t i = 0; i < channels; i++, head++, tail++) {
            if (head < channels)
                add_square_sum(&in[in_shape_.get_index(0, 0, head)], wxh, &in_square_[0]);

            if (tail >= 0)
                sub_square_sum(&in[in_shape_.get_index(0, 0, tail)], wxh, &in_square_[0]);

            float_t *dst = &out[in_shape_.get_index(0, 0, i)];
            const float_t *src = &in[in_shape_.get_index(0, 0, i)];
            for (cnn_size_t j = 0; j < wxh; j++)
                dst[j] = src[j] * std::pow(float_t(1) + alpha_div_size * in_square_[j], -beta_);
        }
    }

    void forward_within(const vec_t& in, vec_t& out) {
        CNN_UNREFERENCED_PARAMETER(in);
        CNN_UNREFERENCED_PARAMETER(out);
        throw nn_error("not implemented");
    }

    void add_square_sum(const float_t *src, cnn_size_t size, float_t *dst) {
        for (cnn_size_t i = 0; i < size; i++)
            dst[i] += src[i] * src[i];
    }

    void sub_square_sum(const float_t *src, cnn_size_t size, float_t *dst) {
        for (cnn_size_t i = 0; i < size; i++)
            dst[i] -= src[i] * src[i];
    }

    layer_shape_t in_shape_;

    cnn_size_t size_;
    float_t alpha_, beta_;
    norm_region region_;

    vec_t in_square_;
};

} // namespace tiny_cnn
