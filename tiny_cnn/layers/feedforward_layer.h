/*
    Copyright (c) 2016, Taiga Nomi
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
#include "tiny_cnn/layers/layer.h"
#include "tiny_cnn/activations/activation_function.h"

namespace tiny_cnn {

/**
 * single-input, single-output network with activation function
 **/
template<typename Activation>
class feedforward_layer : public layer {
public:
    explicit feedforward_layer(const std::vector<vector_type>& in_data_type)
        : layer(in_data_type, std_output_order(true)) {}
    activation::function& activation_function() { return h_; }
    std::pair<float_t, float_t> out_value_range() const override { return h_.scale(); }

protected:

    void backward_activation(const tensor_t& prev_delta, const tensor_t& this_out, tensor_t& curr_delta) {

        // @todo consider parallelism
        for (cnn_size_t sample = 0, sample_count = this_out.size(); sample < sample_count; ++sample) {
            const vec_t& out_vec = this_out[sample];
            const vec_t& prev_delta_vec = prev_delta[sample];
            vec_t& curr_delta_vec = curr_delta[sample];

            const cnn_size_t len = prev_delta_vec.size();
            
            if (h_.one_hot()) {
                for (cnn_size_t c = 0; c < len; c++) {
                    curr_delta_vec[c] = prev_delta_vec[c] * h_.df(out_vec[c]);
                }
            }
            else {
                for (cnn_size_t c = 0; c < len; c++) {
                    vec_t df = h_.df(out_vec, c);
                    curr_delta_vec[c] = vectorize::dot(&prev_delta_vec[0], &df[0], len);
                }
            }
        }
    }

    Activation h_;
};

} // namespace tiny_cnn
