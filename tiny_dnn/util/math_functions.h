// Copyright (c) 2013-2016, Taiga Nomi. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

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
inline void moments(const tensor_t& in, cnn_size_t spatial_dim, cnn_size_t channels, vec_t *mean, vec_t *variance) {
    cnn_size_t num_examples = in.size();

    assert(in[0].size() == spatial_dim * channels);

    mean->resize(channels);
    std::fill(mean->begin(), mean->end(), (float_t)0.0);

    if (variance != nullptr) {
        variance->resize(channels);
        std::fill(variance->begin(), variance->end(), (float_t)0.0);
    }

    // calculate mean
    for (cnn_size_t i = 0; i < num_examples; i++) {
        for (cnn_size_t j = 0; j < channels; j++) {
            float_t*       pmean = &mean->at(j);
            const float_t* X = &in[i][j*spatial_dim];

            for (cnn_size_t k = 0; k < spatial_dim; k++) {
                *pmean += *X++;
            }
        }
    }

    vector_div(*mean, (float_t)num_examples*spatial_dim);

    // calculate variance
    if (variance != nullptr) {
        for (cnn_size_t i = 0; i < num_examples; i++) {
            for (cnn_size_t j = 0; j < channels; j++) {
                float_t* pvar = &variance->at(j);
                const float_t* X = &in[i][j*spatial_dim];
                float_t        EX = (*mean)[j];

                for (cnn_size_t k = 0; k < spatial_dim; k++) {
                    *pvar += pow(*X++ - EX, (float_t)2.0);
                }
            }
        }

        vector_div(*variance, std::max(1.0f, num_examples*spatial_dim-1.0f));
    }
}

}