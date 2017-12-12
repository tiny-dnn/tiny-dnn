/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <vector>

#include "tiny_dnn/core/framework/tensor.h"
#include "tiny_dnn/util/util.h"

namespace tiny_dnn {

// mean-squared-error loss function for regression
class mse {
 public:
  static float_t f(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    float_t d{0.0};

    for (size_t i = 0; i < y.size(); ++i) d += (y[i] - t[i]) * (y[i] - t[i]);

    return d / static_cast<float_t>(y.size());
  }

  static vec_t df(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    vec_t d(t.size());
    float_t factor = float_t(2) / static_cast<float_t>(t.size());

    for (size_t i = 0; i < y.size(); ++i) d[i] = factor * (y[i] - t[i]);

    return d;
  }
};

// absolute loss function for regression
class absolute {
 public:
  static float_t f(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    float_t d{0};

    for (size_t i = 0; i < y.size(); ++i) d += std::abs(y[i] - t[i]);

    return d / static_cast<float_t>(y.size());
  }

  static vec_t df(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    vec_t d(t.size());
    float_t factor = float_t(1) / static_cast<float_t>(t.size());

    for (size_t i = 0; i < y.size(); ++i) {
      float_t sign = y[i] - t[i];
      if (sign < float_t{0.f})
        d[i] = -factor;
      else if (sign > float_t{0.f})
        d[i] = factor;
      else
        d[i] = {0};
    }

    return d;
  }
};

// absolute loss with epsilon range for regression
// epsilon range [-eps, eps] with eps = 1./fraction
template <int fraction>
class absolute_eps {
 public:
  static float_t f(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    float_t d{0};
    const float_t eps = float_t(1) / fraction;

    for (size_t i = 0; i < y.size(); ++i) {
      float_t diff = std::abs(y[i] - t[i]);
      if (diff > eps) d += diff;
    }
    return d / static_cast<float_t>(y.size());
  }

  static vec_t df(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    vec_t d(t.size());
    const float_t factor = float_t(1) / static_cast<float_t>(t.size());
    const float_t eps    = float_t(1) / fraction;

    for (size_t i = 0; i < y.size(); ++i) {
      float_t sign = y[i] - t[i];
      if (sign < -eps)
        d[i] = -factor;
      else if (sign > eps)
        d[i] = factor;
      else
        d[i] = 0.f;
    }
    return d;
  }
};

// cross-entropy loss function for (multiple independent) binary classifications
class cross_entropy {
 public:
  static float_t f(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    float_t d{0};

    for (size_t i = 0; i < y.size(); ++i)
      d += -t[i] * std::log(y[i]) -
           (float_t(1) - t[i]) * std::log(float_t(1) - y[i]);

    return d;
  }

  static vec_t df(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    vec_t d(t.size());

    for (size_t i = 0; i < y.size(); ++i) {
      d[i] = (y[i] - t[i]) / (y[i] * (float_t(1) - y[i]));
    }

    return d;
  }
};

// cross-entropy loss function for multi-class classification
class cross_entropy_multiclass {
 public:
  static float_t f(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    float_t d{0.0};

    for (size_t i = 0; i < y.size(); ++i) d += -t[i] * std::log(y[i]);

    return d;
  }

  static vec_t df(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    vec_t d(t.size());

    for (size_t i = 0; i < y.size(); ++i) d[i] = -t[i] / y[i];

    return d;
  }
};

template <typename E, typename S>
vec_t gradient(const Tensor<float_t, S> &y, const vec_t &t) {
  assert(y.size() == t.size());
  return E::df(y.toVec(), t);
}

template <typename E, typename S1, typename S2>
Tensor<> gradient(const Tensor<float_t, S1> &y, const Tensor<float_t, S2> &t) {
  Tensor<> tmp(y.shape());
  tmp.assign(Tensor<>(E::df(y.toVec(), t.toVec())));
  return tmp;
}

template <typename S1, typename S2>
inline void apply_cost_if_defined(Tensor<float_t, S1> &sample_gradient,
                                  const Tensor<float_t, S2> &sample_cost) {
  if (sample_gradient.shape() == sample_cost.shape()) {
    // @todo consider adding parallelism

    auto it1 = sample_gradient.begin();
    auto it2 = sample_cost.begin();
    for (; it1 < sample_gradient.end(); ++it1, ++it2) {
      *it1 *= *it2;
    }
  }
}

// TODO(Randl): after full Tensor integration, create universal gradient
// function
/**
 * gradient for a minibatch
 * @tparam E
 * @param y
 * @param t
 * @param t_cost
 * @return
 */
template <typename E, typename S1, typename S2, typename S3>
std::vector<Tensor<>> gradient(const Tensor<float_t, S1> &y,
                               const Tensor<float_t, S2> &t,
                               const Tensor<float_t, S3> &t_cost) {
  const size_t sample_count  = y.shape()[0];
  const size_t channel_count = y.shape()[1];

  std::vector<Tensor<>> gradients(y.shape()[0]);

  CNN_UNREFERENCED_PARAMETER(channel_count);
  assert(t_cost.empty() || t_cost.shape() == t.shape());
  assert(t.shape()[0] == sample_count);
  assert(t.shape()[1] == channel_count);

  // @todo add parallelism
  for (size_t sample = 0; sample < sample_count; ++sample) {
    gradients[sample] = gradient<E>(
      y.subView(TensorSingleIndex(sample), TensorAll(), TensorAll()),
      t.subView(TensorSingleIndex(sample), TensorAll(), TensorAll()));

    if (!t_cost.empty() && sample < t_cost.shape()[0]) {
      apply_cost_if_defined(
        gradients[sample],
        t_cost.subView(TensorSingleIndex(sample), TensorAll(), TensorAll()));
    }
  }

  return gradients;
}

}  // namespace tiny_dnn
