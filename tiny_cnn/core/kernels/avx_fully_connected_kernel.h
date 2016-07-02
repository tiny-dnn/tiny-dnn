/*
    Copyright (c) 2016, Taiga Nomi, Edgar Riba
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

#include "tiny_cnn/core/params/fully_params.h"

namespace tiny_cnn {
namespace core {
namespace kernels {

void tiny_fully_connected_kernel(const fully_params& params,
                                 const vec_t&        in,
                                 const vec_t&        W,
                                 vec_t&              b,
                                 vec_t&              a,
                                 const bool          layer_parallelize) {
    for_i(layer_parallelize, params.out_size_, [&](int i) {
        a[i] = float_t(0);
        for (cnn_size_t c = 0; c < params.in_size_; c++) {
            a[i] += W[c * params.out_size_ + i] * in[c];
        }

        if (params.has_bias_) {
            a[i] += b[i];
        }
    });
}

void tiny_fully_connected_back_kernel(const fully_params& params,
                                      const vec_t& prev_out,
                                      const vec_t& W,
                                      vec_t&       dW,
                                      vec_t&       prev_delta,
                                      vec_t&       curr_delta,
                                      vec_t&       db,
                                      const bool   layer_parallelize) {
    for (cnn_size_t c = 0; c < params.in_size_; c++) {
        // propagate delta to previous layer
        // prev_delta[c] += current_delta[r] * W_[c * out_size_ + r]
        prev_delta[c] += vectorize::dot(&curr_delta[0],
                                        &W[c * params.out_size_],
                                        params.out_size_);
    }

    for_(layer_parallelize, 0, size_t(params.out_size_), [&](const blocked_range& r) {
        // accumulate weight-step using delta
        // dW[c * out_size + i] += current_delta[i] * prev_out[c]
        for (cnn_size_t c = 0; c < params.in_size_; c++) {
            vectorize::muladd(&curr_delta[r.begin()],
                              prev_out[c], r.end() - r.begin(),
                              &dW[c * params.out_size_ + r.begin()]);
        }

        if (params.has_bias_) {
            // vec_t& db = *in_grad[2];
            for (int i = r.begin(); i < r.end(); i++) {
                db[i] += curr_delta[i];
            }
        }
    });
}

}  // namespace kernels
}  // namespace core
}  // namespace tiny_cnn
