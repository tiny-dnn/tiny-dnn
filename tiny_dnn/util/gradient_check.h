/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <algorithm>
#include <limits>
#include <vector>
#include "tiny_dnn/layers/layer.h"
#include "tiny_dnn/util/util.h"

namespace tiny_dnn {

/**
 * Computes the numeric gradient of a given layer
 * http://karpathy.github.io/neuralnets/
 * http://cs231n.github.io/neural-networks-3/#gradcheck
 * @param layer Reference to a layer type.
 * @param in_data Input (data, weights, biases, etc).
 * @param in_edge Input edge index to perturb to obtain the gradient (data,
 *weights, biases, etc.)
 * @param in_pos Input position to perturb for retrieving the gradient of a
 *given edge.
 * @param out_data Output matrices (to calculate the increment after
 *perturbation).
 * @param out_grads Output gradients.
 * @param out_edge Output matrix index to calculate the increment after
 *perturbation).
 * @param out_pos Position in the matrix to calculate the increment after
 *perturbation.
 * @return The numeric gradient for the desired position and matrix.
 **/
float_t numeric_gradient(layer &layer,
                         std::vector<tensor_t> in_data,  //  copy is safer
                         const size_t in_edge,
                         const size_t in_pos,
                         std::vector<tensor_t> &out_data,
                         std::vector<tensor_t> &out_grads,
                         const size_t out_edge,
                         const size_t out_pos) {
  // sqrt(machine epsilon) is assumed to be safe
  float_t h = std::sqrt(std::numeric_limits<float_t>::epsilon());
  // initialize input/output
  std::vector<Tensor<>> in_tenses, out_tenses, out_grads_tenses;
  std::vector<Tensor<> *> in_tens_, out_tens_, out_grads_tens_;
  for (auto &t : in_data) {
    in_tenses.push_back(Tensor<>(t));
  }
  for (auto &t : out_data) {
    out_tenses.push_back(Tensor<>(t));
    out_tenses.back().fill(0);
  }
  for (auto &t : out_grads) {
    out_grads_tenses.push_back(Tensor<>(t));
    out_grads_tenses.back().fill(0);
  }

  for (auto &t : in_tenses) {
    in_tens_.push_back(&t);
  }
  for (auto &t : out_tenses) {
    out_tens_.push_back(&t);
  }
  for (auto &t : out_grads_tenses) {
    out_grads_tens_.push_back(&t);
  }

  // Set output gradient to 1 so that input grad is 1*f'(x)
  out_grads_tenses[out_edge].host_at(0, out_pos) = 1.0;
  // Save current input value to perturb
  float_t prev_in = in_tenses[in_edge].host_at(0, in_pos);
  // Perturb by a small amount (-h)
  in_tenses[in_edge].host_at(0, in_pos) = prev_in - h;
  layer.forward_propagation(in_tens_, out_tens_);
  float_t out_1 = out_tens_[out_edge]->host_at(0, out_pos);
  // Perturb by a small amount (+h)
  in_tenses[in_edge].host_at(0, in_pos) = prev_in + h;
  layer.forward_propagation(in_tens_, out_tens_);
  float_t out_2 = out_tens_[out_edge]->host_at(0, out_pos);
  // numerical gradient
  return (out_2 - out_1) / (2 * h);
}

/**
 * Gets the gradient from the implemented backward pass.
 * @param layer Reference to a layer type.
 * @param in_data Input (data, weights, biases, etc).
 * @param in_edge Input edge index for retrieving the gradient (data, weights,
 * biases, etc.)
 * @param in_pos Input position for retrieving the gradient.
 * @param out_data Output data matrices.
 * @param out_grads Next layer gradients (will be 1 for the tested position).
 * @param out_edge Output matrix to put the gradient to 1.
 * @param out_pos Output position to put the gradient to 1.
 * @return The computed gradient for the desired position and matrix.
 */
float_t analytical_gradient(layer &layer,
                            std::vector<tensor_t> in_data,
                            const size_t in_edge,
                            const size_t in_pos,
                            std::vector<tensor_t> &out_data,
                            std::vector<tensor_t> &out_grads,
                            const size_t out_edge,
                            const size_t out_pos) {
  // initialize input/output
  std::vector<Tensor<>> in_tenses, in_grads_tenses, out_tenses,
    out_grads_tenses;
  std::vector<Tensor<> *> in_tens_, in_grads_tens_, out_tens_, out_grads_tens_;
  for (auto &t : in_data) {
    in_tenses.push_back(Tensor<>(t));
  }
  in_grads_tenses = in_tenses;
  for (auto &t : out_data) {
    out_tenses.push_back(Tensor<>(t));
    out_tenses.back().fill(0);
  }
  for (auto &t : out_grads) {
    out_grads_tenses.push_back(Tensor<>(t));
    out_grads_tenses.back().fill(0);
  }

  for (auto &t : in_tenses) {
    in_tens_.push_back(&t);
  }
  for (auto &t : in_grads_tenses) {
    in_grads_tens_.push_back(&t);
  }
  for (auto &t : out_tenses) {
    out_tens_.push_back(&t);
  }
  for (auto &t : out_grads_tenses) {
    out_grads_tens_.push_back(&t);
  }

  out_grads_tenses[out_edge].host_at(0, out_pos) =
    1.0;  // set target grad to 1.
  // get gradient by plain backpropagation
  layer.forward_propagation(in_tens_, out_tens_);
  layer.back_propagation(in_tens_, out_tens_, out_grads_tens_, in_grads_tens_);
  return in_grads_tenses[in_edge].host_at(0, in_pos);
}

/**
 * Calculates the relative error between the real and the numeric gradient.
 * |d1 - d2| / max(|d1|, |d2|)
 * http://cs231n.github.io/neural-networks-3/#gradcheck
 * @param analytical_gradient
 * @param numeric_gradient
 * @return the relative error.
 */
float_t relative_error(const float_t analytical_grad,
                       const float_t numeric_grad) {
  float_t max = std::max(std::abs(analytical_grad), std::abs(numeric_grad));
  return (max == 0) ? 0.0 : std::abs(analytical_grad - numeric_grad) / max;
}

}  // namespace tiny_dnn
