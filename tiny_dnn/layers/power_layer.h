/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "tiny_dnn/layers/layer.h"
#include "tiny_dnn/util/util.h"

namespace tiny_dnn {

/**
 * element-wise pow: ```y = scale*x^factor```
 **/
class power_layer : public layer {
 public:
  typedef layer Base;

  /**
   * @param in_shape [in] shape of input tensor
   * @param factor   [in] floating-point number that specifies a power
   * @param scale    [in] scale factor for additional multiply
   */
  power_layer(const shape3d &in_shape,
              float_t factor,
              float_t scale = float_t{1.0})
    : layer({vector_type::data}, {vector_type::data}),
      in_shape_(in_shape),
      factor_(factor),
      scale_(scale) {}

  /**
   * @param prev_layer [in] previous layer to be connected
   * @param factor     [in] floating-point number that specifies a power
   * @param scale      [in] scale factor for additional multiply
   */
  power_layer(const layer &prev_layer,
              float_t factor,
              float_t scale = float_t{1.0})
    : layer({vector_type::data}, {vector_type::data}),
      in_shape_(prev_layer.out_shape()[0]),
      factor_(factor),
      scale_(scale) {}

  std::string layer_type() const override { return "power"; }

  std::vector<shape3d> in_shape() const override { return {in_shape_}; }

  std::vector<shape3d> out_shape() const override { return {in_shape_}; }

  void forward_propagation(const std::vector<tensor_t *> &in_data,
                           std::vector<tensor_t *> &out_data) override {
    const tensor_t &x = *in_data[0];
    tensor_t &y       = *out_data[0];

    for (size_t i = 0; i < x.size(); i++) {
      std::transform(x[i].begin(), x[i].end(), y[i].begin(),
                     [=](float_t x) { return scale_ * std::pow(x, factor_); });
    }
  }

  void back_propagation(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) override {
    tensor_t &dx       = *in_grad[0];
    const tensor_t &dy = *out_grad[0];
    const tensor_t &x  = *in_data[0];
    const tensor_t &y  = *out_data[0];

    for (size_t i = 0; i < x.size(); i++) {
      for (size_t j = 0; j < x[i].size(); j++) {
        // f(x) = (scale*x)^factor
        // ->
        //   dx = dy * df(x)
        //      = dy * scale * factor * (scale * x)^(factor - 1)
        //      = dy * scale * factor * (scale * x)^factor * (scale *
        //      x)^(-1)
        //      = dy * factor * y / x
        if (std::abs(x[i][j]) > 1e-10) {
          dx[i][j] = dy[i][j] * factor_ * y[i][j] / x[i][j];
        } else {
          dx[i][j] = dy[i][j] * scale_ * factor_ *
                     std::pow(scale_ * x[i][j], factor_ - 1.0);
        }
      }
    }
  }

  float_t factor() const { return factor_; }

  float_t scale() const { return scale_; }

  friend struct serialization_buddy;

 private:
  shape3d in_shape_;
  float_t factor_;
  float_t scale_;
};

}  // namespace tiny_dnn
