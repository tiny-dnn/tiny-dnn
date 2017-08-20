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

  void forward_propagation(const std::vector<Tensor<> *> &in_data,
                           std::vector<Tensor<> *> &out_data) override {
    const size_t sample_count = in_data[0]->shape()[0];

    for (size_t i = 0; i < sample_count; i++) {
      auto x = in_data[0]->subView(TensorSingleIndex(i), TensorAll());
      auto y = out_data[0]->subView(TensorSingleIndex(i), TensorAll());
      std::transform(x.host_begin(), x.host_end(), y.host_begin(),
                     [=](float_t z) { return scale_ * std::pow(z, factor_); });
    }
  }

  void back_propagation(const std::vector<Tensor<> *> &in_data,
                        const std::vector<Tensor<> *> &out_data,
                        std::vector<Tensor<> *> &out_grad,
                        std::vector<Tensor<> *> &in_grad) override {
    Tensor<> &dx       = *in_grad[0];
    const Tensor<> &dy = *out_grad[0];
    const Tensor<> &x  = *in_data[0];
    const Tensor<> &y  = *out_data[0];

    const size_t sample_count = x.shape()[0], num = x.shape()[1];

    for (size_t i = 0; i < sample_count; ++i) {
      for (size_t j = 0; j < num; ++j) {
        // f(x) = (scale*x)^factor
        // ->
        //   dx = dy * df(x)
        //      = dy * scale * factor * (scale * x)^(factor - 1)
        //      = dy * scale * factor * (scale * x)^factor * (scale *
        //      x)^(-1)
        //      = dy * factor * y / x
        if (std::abs(x.host_at(i, j)) > 1e-10) {
          dx.host_at(i, j) =
            dy.host_at(i, j) * factor_ * y.host_at(i, j) / x.host_at(i, j);
        } else {
          dx.host_at(i, j) = dy.host_at(i, j) * scale_ * factor_ *
                             std::pow(x.host_at(i, j), factor_ - 1.0f);
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
