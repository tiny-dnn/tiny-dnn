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

    void backward_activation(const fvec_t& prev_delta, const fvec_t& this_out, fvec_t& curr_delta) {
        if (h_.one_hot()) {
#ifdef CNN_USE_AVX
			cnn_size_t sz = prev_delta.size();
			cnn_size_t nblocks = sz >> 3;
			const float* prev = &prev_delta[0];
			const float* out = &this_out[0];
			float* curr = &curr_delta[0];
			for (cnn_size_t i=0; i<nblocks; ++i) {
				__m256 p = _mm256_load_ps(prev);
				__m256 o = _mm256_load_ps(out);
				o = h_.df_ps(o);
				__m256 c = _mm256_mul_ps(p, o);
				_mm256_store_ps(curr, c);
				prev += 8;
				out += 8;
				curr += 8;
			}
			for (cnn_size_t c=nblocks<<3; c<sz; ++c) {
                curr_delta[c] = prev_delta[c] * h_.df(this_out[c]);
			}
#else
            for (cnn_size_t c = 0; c < prev_delta.size(); c++) {
                curr_delta[c] = prev_delta[c] * h_.df(this_out[c]);
            }
#endif
        }
        else {
            for (cnn_size_t c = 0; c < prev_delta.size(); c++) {
                vec_t df = h_.df(this_out, c);
                curr_delta[c] = vectorize::dot(&prev_delta[0], &df[0], prev_delta.size());
            }
        }
    }

    void backward_activation(const dvec_t& prev_delta, const dvec_t& this_out, dvec_t& curr_delta) {
        if (h_.one_hot()) {
            for (cnn_size_t c = 0; c < prev_delta.size(); c++) {
                curr_delta[c] = prev_delta[c] * h_.df(this_out[c]);
            }
        }
        else {
            for (cnn_size_t c = 0; c < prev_delta.size(); c++) {
                vec_t df = h_.df(this_out, c);
                curr_delta[c] = vectorize::dot(&prev_delta[0], &df[0], prev_delta.size());
            }
		}
	}

    Activation h_;
};

} // namespace tiny_cnn
