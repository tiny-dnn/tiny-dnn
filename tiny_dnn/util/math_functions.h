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
#include <algorithm>

#include "tiny_dnn/util/util.h"

namespace tiny_dnn {

// x = x / denom
inline void vector_div(vec_t& x, float_t denom) {
    std::transform(x.begin(), x.end(), x.begin(), [=](float_t x) { return x / denom; });
}

/** 
 * calculate mean/variance across channels
 */
inline void moments(const tensor_t& in, serial_size_t spatial_dim, serial_size_t channels, vec_t *mean, vec_t *variance) {
    serial_size_t num_examples = static_cast<serial_size_t>(in.size());

    assert(in[0].size() == spatial_dim * channels);

    mean->resize(channels);
    std::fill(mean->begin(), mean->end(), (float_t)0.0);

    if (variance != nullptr) {
        variance->resize(channels);
        std::fill(variance->begin(), variance->end(), (float_t)0.0);
    }

    // calculate mean
    for (serial_size_t i = 0; i < num_examples; i++) {
        for (serial_size_t j = 0; j < channels; j++) {
            float_t*       pmean = &mean->at(j);
            const float_t* X = &in[i][j*spatial_dim];

            for (serial_size_t k = 0; k < spatial_dim; k++) {
                *pmean += *X++;
            }
        }
    }

    vector_div(*mean, (float_t)num_examples*spatial_dim);

    // calculate variance
    if (variance != nullptr) {
        for (serial_size_t i = 0; i < num_examples; i++) {
            for (serial_size_t j = 0; j < channels; j++) {
                float_t* pvar = &variance->at(j);
                const float_t* X = &in[i][j*spatial_dim];
                float_t        EX = (*mean)[j];

                for (serial_size_t k = 0; k < spatial_dim; k++) {
                    *pvar += pow(*X++ - EX, (float_t)2.0);
                }
            }
        }

        vector_div(*variance, std::max(1.0f, num_examples*spatial_dim-1.0f));
    }
}

}