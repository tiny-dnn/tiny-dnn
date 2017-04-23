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

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include <algorithm>
#include <numeric>

#include "tiny_dnn/util/util.h"

namespace tiny_dnn {

// x = x / denom
inline void vector_div(vec_t &x, float_t denom) {
  std::transform(x.begin(), x.end(), x.begin(),
                 [=](float_t x) { return x / denom; });
}

namespace detail {

inline void moments_impl_calc_mean(size_t num_examples,
                                   size_t channels,
                                   size_t spatial_dim,
                                   const tensor_t &in,
                                   vec_t &mean) {
  for (size_t i = 0; i < num_examples; i++) {
    for (size_t j = 0; j < channels; j++) {
      float_t &rmean = mean.at(j);
      const auto it  = in[i].begin() + (j * spatial_dim);
      rmean          = std::accumulate(it, it + spatial_dim, rmean);
    }
  }
}

inline void moments_impl_calc_variance(size_t num_examples,
                                       size_t channels,
                                       size_t spatial_dim,
                                       const tensor_t &in,
                                       const vec_t &mean,
                                       vec_t &variance) {
  assert(mean.size() >= channels);
  for (size_t i = 0; i < num_examples; i++) {
    for (size_t j = 0; j < channels; j++) {
      float_t &rvar    = variance[j];
      const auto it    = in[i].begin() + (j * spatial_dim);
      const float_t ex = mean[j];
      rvar             = std::accumulate(it, it + spatial_dim, rvar,
                             [ex](float_t current, float_t x) {
                               return current + pow(x - ex, float_t{2.0});
                             });
    }
  }
  vector_div(
    variance,
    std::max(float_t{1.0f},
             static_cast<float_t>(num_examples * spatial_dim) - float_t{1.0f}));
}

}  // namespace detail

/**
 * calculate mean/variance across channels
 */
inline void moments(const tensor_t &in,
                    size_t spatial_dim,
                    size_t channels,
                    vec_t &mean) {
  const size_t num_examples = static_cast<serial_size_t>(in.size());
  assert(in[0].size() == spatial_dim * channels);

  mean.resize(channels);
  vectorize::fill(&mean[0], mean.size(), float_t{0.0});
  detail::moments_impl_calc_mean(num_examples, channels, spatial_dim, in, mean);
  vector_div(mean, (float_t)num_examples * spatial_dim);
}

inline void moments(const tensor_t &in,
                    size_t spatial_dim,
                    size_t channels,
                    vec_t &mean,
                    vec_t &variance) {
  const size_t num_examples = static_cast<serial_size_t>(in.size());
  assert(in[0].size() == spatial_dim * channels);

  // calc mean
  moments(in, spatial_dim, channels, mean);

  variance.resize(channels);
  vectorize::fill(&variance[0], variance.size(), float_t{0.0});
  detail::moments_impl_calc_variance(num_examples, channels, spatial_dim, in,
                                     mean, variance);
}

}  // namespace tiny_dnn
