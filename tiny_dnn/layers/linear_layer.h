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
#include <algorithm>


namespace tiny_dnn {

/**
 * element-wise operation: ```f(x) = h(scale*x+bias)```
 */
template<typename Activation>
class linear_layer : public feedforward_layer<Activation> {
public:
    CNN_USE_LAYER_MEMBERS;

    typedef feedforward_layer<Activation> Base;

    /**
     * @param dim   [in] number of elements
     * @param scale [in] factor by which to multiply
     * @param bias  [in] bias term
     **/
    explicit linear_layer(serial_size_t dim, float_t scale = float_t(1), float_t bias = float_t(0))
        : Base({vector_type::data}),
        dim_(dim), scale_(scale), bias_(bias) {}

    std::vector<shape3d> in_shape() const override {
        return {shape3d(dim_, 1, 1) };
    }

    std::vector<shape3d> out_shape() const override {
        return{ shape3d(dim_, 1, 1), shape3d(dim_, 1, 1) };
    }

    std::string layer_type() const override { return "linear"; }

    void forward_propagation(const std::vector<tensor_t*>& in_data,
                             std::vector<tensor_t*>& out_data) override {
        const tensor_t& in  = *in_data[0];
        tensor_t&       out = *out_data[0];
        tensor_t&       a   = *out_data[1];

        // do nothing
        CNN_UNREFERENCED_PARAMETER(out);

        // @todo revise the parallelism strategy
        for_i(parallelize_, dim_, [&](int i) {
            for (serial_size_t sample = 0, sample_count = static_cast<serial_size_t>(in.size()); sample < sample_count; ++sample)
                a[sample][i] = scale_ * in[sample][i] + bias_;
        });
        this->forward_activation(*out_data[0], *out_data[1]);
    }

    void back_propagation(const std::vector<tensor_t*>& in_data,
                          const std::vector<tensor_t*>& out_data,
                          std::vector<tensor_t*>&       out_grad,
                          std::vector<tensor_t*>&       in_grad) override {
        tensor_t& prev_delta = *in_grad[0];
        tensor_t& curr_delta = *out_grad[1];

        CNN_UNREFERENCED_PARAMETER(in_data);

        this->backward_activation(*out_grad[0], *out_data[0], curr_delta);

        // @todo revise parallelism strategy
        for (serial_size_t sample = 0; sample < static_cast<serial_size_t>(prev_delta.size()); ++sample) {
            for_i(parallelize_, dim_, [&](int i) {
                prev_delta[sample][i] = curr_delta[sample][i] * scale_;
            });
        }
    }

    template <class Archive>
    static void load_and_construct(Archive & ar, cereal::construct<linear_layer> & construct) {
        serial_size_t dim;
        float_t scale, bias;

        ar(cereal::make_nvp("in_size", dim), cereal::make_nvp("scale", scale), cereal::make_nvp("bias", bias));

        construct(dim, scale, bias);
    }

    template <class Archive>
    void serialize(Archive & ar) {
        layer::serialize_prolog(ar);
        ar(cereal::make_nvp("in_size", dim_), cereal::make_nvp("scale", scale_), cereal::make_nvp("bias", bias_));
    }

protected:
    serial_size_t dim_;
    float_t scale_, bias_;
};

} // namespace tiny_dnn
