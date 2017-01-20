/*
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once
#include <algorithm>
#include <utility>
#include "tiny_dnn/util/util.h"

namespace tiny_dnn {
namespace activation {

class function {
 public:
  function()                 = default;
  function(const function &) = default;
#ifndef CNN_DEFAULT_MOVE_CONSTRUCTOR_UNAVAILABLE
  function(function &&) = default;
#endif
  function &operator=(const function &) = default;
#ifndef CNN_DEFAULT_ASSIGNMENT_OPERATOR_UNAVAILABLE
  function &operator=(function &&) = default;
#endif
  virtual ~function() = default;

  virtual float_t f(const vec_t &v, size_t index) const = 0;
  void itef(vec_t &out, const vec_t &in, size_t cnt) const {
    for (size_t i = 0; i < cnt; i++) {
      out[i] = f(in, i);
    }
  }

  // dfi/dyi
  virtual float_t df(float_t y) const = 0;
  void itedf(vec_t &cur,
             const vec_t &prev,
             const vec_t &out,
             size_t cnt) const {
    for (size_t i = 0; i < cnt; i++) {
      cur[i] = prev[i] * df(out[i]);
    }
  }

  // dfi/dyk (k=0,1,..n)
  virtual vec_t df(const vec_t &y, size_t i) const {
    vec_t v(y.size(), 0);
    v[i] = df(y[i]);
    return v;
  }

  // return if dfi/dyk is one-hot vector
  virtual bool one_hot() const { return true; }

  // target value range for learning
  virtual std::pair<float_t, float_t> scale() const = 0;
};

class identity : public function {
 public:
  using function::df;
  inline float_t f(const vec_t &v, size_t i) const override { return v[i]; }
  inline void itef(vec_t &out, const vec_t &in, size_t cnt) const {
    CNN_UNREFERENCED_PARAMETER(cnt);
    out = in;
  }

  inline float_t df(float_t /*y*/) const override { return float_t(1.0); }
  void itedf(vec_t &cur,
             const vec_t &prev,
             const vec_t &out,
             size_t cnt) const {
    CNN_UNREFERENCED_PARAMETER(out);
    CNN_UNREFERENCED_PARAMETER(cnt);
    cur = prev;
  }
  std::pair<float_t, float_t> scale() const override {
    return std::make_pair(float_t(0.1), float_t(0.9));
  }
};

class sigmoid : public function {
 public:
  using function::df;
  float_t f(const vec_t &v, size_t i) const override {
    return float_t(1) / (float_t(1) + std::exp(-v[i]));
  }
  float_t df(float_t y) const override { return y * (float_t(1) - y); }
  std::pair<float_t, float_t> scale() const override {
    return std::make_pair(float_t(0.1), float_t(0.9));
  }
};

class relu : public function {
 public:
  using function::df;
  float_t f(const vec_t &v, size_t i) const override {
    return std::max(float_t{0}, v[i]);
  }
  float_t df(float_t y) const override {
    return y > float_t{0} ? float_t{1} : float_t{0};
  }
  std::pair<float_t, float_t> scale() const override {
    return std::make_pair(float_t(0.1), float_t(0.9));
  }
};

typedef relu rectified_linear;  // for compatibility

class leaky_relu : public function {
 public:
  using function::df;
  float_t f(const vec_t &v, size_t i) const override {
    return (v[i] > float_t{0}) ? v[i] : float_t(0.01) * v[i];
  }
  float_t df(float_t y) const override {
    return y > float_t{0} ? float_t{1} : float_t(0.01);
  }
  std::pair<float_t, float_t> scale() const override {
    return std::make_pair(float_t(0.1), float_t(0.9));
  }
};

class elu : public function {
 public:
  using function::df;
  float_t f(const vec_t &v, size_t i) const override {
    return (v[i] < float_t(0) ? (exp(v[i]) - float_t(1)) : v[i]);
  }
  float_t df(float_t y) const override {
    return (y > float_t(0) ? float_t(1) : (float_t(1) + y));
  }
  std::pair<float_t, float_t> scale() const override {
    return std::make_pair(float_t(0.1), float_t(0.9));
  }
};

class softmax : public function {
 public:
  float_t f(const vec_t &v, size_t i) const override {
    float_t alpha = *std::max_element(v.begin(), v.end());
    float_t numer = std::exp(v[i] - alpha);
    float_t denom{0};
    for (auto x : v) denom += std::exp(x - alpha);
    return numer / denom;
  }
  void itef(vec_t &out, const vec_t &in, size_t cnt) const {
    float_t alpha = *std::max_element(in.begin(), in.end());
    for (size_t i = 0; i < cnt; i++) {
      float_t numer = std::exp(in[i] - alpha);
      float_t denom{0};
      for (auto x : in) denom += std::exp(x - alpha);
      out[i] = numer / denom;
    }
  }

  float_t df(float_t y) const override { return y * (float_t(1) - y); }

  vec_t df(const vec_t &y, size_t index) const override {
    vec_t v(y.size(), 0);
    for (size_t i = 0; i < y.size(); i++)
      v[i]        = (i == index) ? df(y[index]) : -y[i] * y[index];

    return v;
  }

  bool one_hot() const override { return false; }

  std::pair<float_t, float_t> scale() const override {
    return std::make_pair(float_t{0}, float_t{1});
  }
};

class tan_h : public function {
 public:
  using function::df;

  inline float_t f(const vec_t &v, size_t i) const override {
    return std::tanh(v[i]);
  }

  inline void itef(vec_t &out, const vec_t &in, size_t cnt) const {
    for (size_t i = 0; i < cnt; i++) {
      out[i] = std::tanh(in[i]);
    }
  }

  // fast approximation of tanh (improve 2-3% speed in LeNet-5)
  /*float_t f(float_t x) const {
      const float_t x2 = x * x;
      x *= 1.0 + x2 * (0.1653 + x2 * 0.0097);
      return x / std::sqrt(1.0 + x * x);// invsqrt(static_cast<float>(1.0 + x
  *
  x));
  }*/

  inline float_t df(float_t y) const override { return float_t(1) - sqr(y); }
  void itedf(vec_t &cur,
             const vec_t &prev,
             const vec_t &out,
             size_t cnt) const {
    for (size_t i = 0; i < cnt; i++) {
      cur[i] = prev[i] * df(out[i]);
    }
  }

  std::pair<float_t, float_t> scale() const override {
    return std::make_pair(float_t(-0.8), float_t(0.8));
  }

 private:
  /*float invsqrt(float x) const {
      float x2 = x * 0.5f;
      long i = *reinterpret_cast<long*>(&x);

      i = 0x5f3759df - (i >> 1);
      x = *reinterpret_cast<float*>(&i);
      x = x * (1.5f - (x2 * x * x));
      return x;
  }*/
};

// s tan_h, but scaled to match the other functions
class tan_hp1m2 : public function {
 public:
  using function::df;
  float_t f(const vec_t &v, size_t i) const override {
    const float_t ep = std::exp(v[i]);
    return ep / (ep + std::exp(-v[i]));
  }

  float_t df(float_t y) const override { return 2 * y * (float_t(1) - y); }
  std::pair<float_t, float_t> scale() const override {
    return std::make_pair(float_t(0.1), float_t(0.9));
  }
};

}  // namespace activation
}  // namespace tiny_dnn
