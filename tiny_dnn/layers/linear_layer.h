/*
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <algorithm>

#include "tiny_dnn/util/util.h"

namespace tiny_dnn {

/**
 * element-wise operation: ```f(x) = h(scale*x+bias)```
 */
template <typename Activation>
class linear_layer : public feedforward_layer<Activation> {
 public:
  CNN_USE_LAYER_MEMBERS;

  typedef feedforward_layer<Activation> Base;

  /**
   * @param dim   [in] number of elements
   * @param scale [in] factor by which to multiply
   * @param bias  [in] bias term
   **/
  explicit linear_layer(serial_size_t dim,
                        float_t scale = float_t{1},
                        float_t bias  = float_t{0})
    : Base({vector_type::data}), dim_(dim), scale_(scale), bias_(bias) {}

  std::vector<shape3d> in_shape() const override {
    return {shape3d(dim_, 1, 1)};
  }

  std::vector<shape3d> out_shape() const override {
    return {shape3d(dim_, 1, 1), shape3d(dim_, 1, 1)};
  }

  std::string layer_type() const override { return "linear"; }

  void forward_propagation(const std::vector<tensor_t *> &in_data,
                           std::vector<tensor_t *> &out_data) override {
    const tensor_t &in = *in_data[0];
    tensor_t &out      = *out_data[0];
    tensor_t &a        = *out_data[1];

    // do nothing
    CNN_UNREFERENCED_PARAMETER(out);

    // @todo revise the parallelism strategy
    for_i(parallelize_, dim_, [&](int i) {
      for (serial_size_t sample       = 0,
                         sample_count = static_cast<serial_size_t>(in.size());
           sample < sample_count; ++sample)
        a[sample][i] = scale_ * in[sample][i] + bias_;
    });
    this->forward_activation(*out_data[0], *out_data[1]);
  }

  void back_propagation(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) override {
    tensor_t &prev_delta = *in_grad[0];
    tensor_t &curr_delta = *out_grad[1];

    CNN_UNREFERENCED_PARAMETER(in_data);

    this->backward_activation(*out_grad[0], *out_data[0], curr_delta);

    // @todo revise parallelism strategy
    for (serial_size_t sample = 0;
         sample < static_cast<serial_size_t>(prev_delta.size()); ++sample) {
      for_i(parallelize_, dim_, [&](int i) {
        prev_delta[sample][i] = curr_delta[sample][i] * scale_;
      });
    }
  }

#ifndef CNN_NO_SERIALIZATION
  friend struct serialization_buddy;
#endif

 protected:
  serial_size_t dim_;
  float_t scale_, bias_;
};

}  // namespace tiny_dnn
