/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <unordered_map>

#include "tiny_dnn/util/util.h"

namespace tiny_dnn {

/**
 * base class of optimizer
 * usesHessian : true if an optimizer uses hessian (2nd order derivative of loss
 *function)
 **/
struct optimizer {
  optimizer()                  = default;
  optimizer(const optimizer &) = default;
  optimizer(optimizer &&)      = default;
  optimizer &operator=(const optimizer &) = default;
  optimizer &operator=(optimizer &&) = default;
  virtual ~optimizer()               = default;
  virtual void update(const vec_t &dW, Tensor<> &W, bool parallelize) = 0;
  virtual void reset() {}  // override to implement pre-learning action
};

// helper class to hold N values for each weight
template <int N>
struct stateful_optimizer : public optimizer {
  void reset() override {
    for (auto &e : E_) e.clear();
  }

 protected:
  template <int Index>
  Tensor<> &get(const Tensor<> &key) {
    static_assert(Index < N, "index out of range");
    if (E_[Index][&key].empty()) {
      E_[Index][&key].reshape(key.shape());
      E_[Index][&key].fill(0);
    }
    return E_[Index][&key];
  }

  std::unordered_map<const Tensor<> *, Tensor<> > E_[N];
};

/**
 * adaptive gradient method
 *
 * J Duchi, E Hazan and Y Singer,
 * Adaptive subgradient methods for online learning and stochastic optimization
 * The Journal of Machine Learning Research, pages 2121-2159, 2011.
 **/
struct adagrad : public stateful_optimizer<1> {
  adagrad() : alpha(float_t(0.01)), eps(float_t(1e-8)) {}

  void update(const vec_t &dW, Tensor<> &W, bool parallelize) {
    Tensor<> &g = get<0>(W);
    for_i(parallelize, W.size(), [&](size_t i) {
      g.host_at(i) += dW[i] * dW[i];
      W.host_at(i) -= alpha * dW[i] / (std::sqrt(g.host_at(i)) + eps);
    });
  }

  float_t alpha;  // learning rate
 private:
  float_t eps;
};

/**
 * RMSprop
 *
 * T Tieleman, and G E Hinton,
 * Lecture 6.5 - rmsprop, COURSERA: Neural Networks for Machine Learning (2012)
 **/
struct RMSprop : public stateful_optimizer<1> {
  RMSprop() : alpha(float_t(0.0001)), mu(float_t(0.99)), eps(float_t(1e-8)) {}

  void update(const vec_t &dW, Tensor<> &W, bool parallelize) {
    Tensor<> &g = get<0>(W);

    for_i(parallelize, W.size(), [&](size_t i) {
      g.host_at(i) = mu * g.host_at(i) + (1 - mu) * dW[i] * dW[i];
      W.host_at(i) -= alpha * dW[i] / std::sqrt(g.host_at(i) + eps);
    });
  }

  float_t alpha;  // learning rate
  float_t mu;     // decay term
 private:
  float_t eps;  // constant value to avoid zero-division
};

/**
 * @brief [a new optimizer (2015)]
 * @details [see Adam: A Method for Stochastic Optimization (Algorithm 1)
 *               http://arxiv.org/abs/1412.6980]
 *
 */
struct adam : public stateful_optimizer<2> {
  adam()
    : alpha(float_t(0.001)),
      b1(float_t(0.9)),
      b2(float_t(0.999)),
      b1_t(float_t(0.9)),
      b2_t(float_t(0.999)),
      eps(float_t(1e-8)) {}

  void update(const vec_t &dW, Tensor<> &W, bool parallelize) {
    Tensor<> &mt = get<0>(W);
    Tensor<> &vt = get<1>(W);

    b1_t *= b1;
    b2_t *= b2;

    for_i(parallelize, W.size(), [&](size_t i) {
      mt.host_at(i) = b1 * mt.host_at(i) + (float_t(1) - b1) * dW[i];
      vt.host_at(i) = b2 * vt.host_at(i) + (float_t(1) - b2) * dW[i] * dW[i];

      W.host_at(i) -= alpha * (mt.host_at(i) / (float_t(1) - b1_t)) /
                      std::sqrt((vt.host_at(i) / (float_t(1) - b2_t)) + eps);
    });
  }

  float_t alpha;  // learning rate
  float_t b1;     // decay term
  float_t b2;     // decay term
  float_t b1_t;   // decay term power t
  float_t b2_t;   // decay term power t

 private:
  float_t eps;  // constant value to avoid zero-division
};

/**
 * SGD without momentum
 *
 * slightly faster than tiny_dnn::momentum
 **/
struct gradient_descent : public optimizer {
  gradient_descent() : alpha(float_t(0.01)), lambda(float_t(0)) {}

  void update(const vec_t &dW, Tensor<> &W, bool parallelize) {
    for_i(parallelize, W.size(), [&](size_t i) {
      W.host_at(i) = W.host_at(i) - alpha * (dW[i] + lambda * W.host_at(i));
    });
  }

  float_t alpha;   // learning rate
  float_t lambda;  // weight decay
};

/**
 * SGD with momentum
 *
 * B T Polyak,
 * Some methods of speeding up the convergence of iteration methods
 * USSR Computational Mathematics and Mathematical Physics, 4(5):1-17, 1964.
 **/
struct momentum : public stateful_optimizer<1> {
 public:
  momentum() : alpha(float_t(0.01)), lambda(float_t(0)), mu(float_t(0.9)) {}

  void update(const vec_t &dW, Tensor<> &W, bool parallelize) {
    Tensor<> &dWprev = get<0>(W);

    for_i(parallelize, W.size(), [&](size_t i) {
      float_t V =
        mu * dWprev.host_at(i) - alpha * (dW[i] + W.host_at(i) * lambda);
      W.host_at(i) += V;
      dWprev.host_at(i) = V;
    });
  }

  float_t alpha;   // learning rate
  float_t lambda;  // weight decay
  float_t mu;      // momentum
};

/**
 * SGD with Nesterov momentum
 *
 * Y Nesterov,
 * A method for unconstrained convex minimization problem with the rate of
 * convergence o(1/k2), Doklady ANSSSR, vol.269, pp.543-547, 1983.
 **/
struct nesterov_momentum : public stateful_optimizer<1> {
 public:
  nesterov_momentum()
    : alpha(float_t(0.01)), lambda(float_t(0)), mu(float_t(0.9)) {}

  void update(const vec_t &dW, Tensor<> &W, bool parallelize) {
    Tensor<> &dWprev = get<0>(W);

    for_i(parallelize, W.size(), [&](size_t i) {
      float_t V =
        mu * dWprev.host_at(i) - alpha * (dW[i] + W.host_at(i) * lambda);
      W.host_at(i) += (-mu) * dWprev.host_at(i) + (1 + mu) * V;
      dWprev.host_at(i) = V;
    });
  }

  float_t alpha;   // learning rate
  float_t lambda;  // weight decay
  float_t mu;      // momentum
};

/**
 * SGD with Nesterov momentum
 *
 * Y Nesterov,
 * A method for unconstrained convex minimization problem with the rate of
 * convergence o(1/k2), Doklady ANSSSR, vol.269, pp.543-547, 1983.
 **/
struct nesterov_momentum : public stateful_optimizer<1> {
 public:
  nesterov_momentum()
    : alpha(float_t(0.01)), lambda(float_t(0)), mu(float_t(0.9)) {}

  void update(const vec_t &dW, vec_t &W, bool parallelize) {
    vec_t &dWprev = get<0>(W);

    for_i(parallelize, W.size(), [&](size_t i) {
      float_t V = mu * dWprev[i] - alpha * (dW[i] + W[i] * lambda);
      W[i] += (-mu) * dWprev[i] + (1 + mu) * V;
      dWprev[i] = V;
    });
  }

  float_t alpha;   // learning rate
  float_t lambda;  // weight decay
  float_t mu;      // momentum
};

}  // namespace tiny_dnn
