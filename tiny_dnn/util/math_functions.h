/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
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
  const size_t num_examples = in.size();
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
  const size_t num_examples = in.size();
  assert(in[0].size() == spatial_dim * channels);

  // calc mean
  moments(in, spatial_dim, channels, mean);

  variance.resize(channels);
  vectorize::fill(&variance[0], variance.size(), float_t{0.0});
  detail::moments_impl_calc_variance(num_examples, channels, spatial_dim, in,
                                     mean, variance);
}

}  // namespace tiny_dnn
