/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "tiny_dnn/layers/layer.h"
#include "tiny_dnn/util/util.h"

namespace tiny_dnn {

/**
 * element-wise add N vectors ```y_i = x0_i + x1_i + ... + xnum_i```
 **/
class elementwise_add_layer : public layer {
 public:
  /**
   * @param num_args [in] number of inputs
   * @param dim      [in] number of elements for each input
   */
  elementwise_add_layer(size_t num_args, size_t dim)
    : layer(std::vector<vector_type>(num_args, vector_type::data),
            {vector_type::data}),
      num_args_(num_args),
      dim_(dim) {}

  std::string layer_type() const override { return "elementwise-add"; }

  std::vector<shape3d> in_shape() const override {
    return std::vector<shape3d>(num_args_, shape3d(dim_, 1, 1));
  }

  std::vector<shape3d> out_shape() const override {
    return {shape3d(dim_, 1, 1)};
  }

  void forward_propagation(const std::vector<Tensor<> *> &in_data,
                           std::vector<Tensor<> *> &out_data) override {
    out_data[0]->fill(0);
    // TODO(Randl): parallelize
    for (size_t sample = 0; sample < in_data[0]->shape()[0]; ++sample) {
      for (size_t i = 0; i < num_args_; i++) {
        auto in_s = in_data[i]->subView(TensorSingleIndex(sample), TensorAll());
        auto out_s =
          out_data[0]->subView(TensorSingleIndex(sample), TensorAll());
        std::transform(in_s.host_begin(), in_s.host_end(), out_s.host_begin(),
                       out_s.host_begin(),
                       [](float_t x, float_t y) { return x + y; });
      }
    }
  }

  void back_propagation(const std::vector<Tensor<> *> &in_data,
                        const std::vector<Tensor<> *> &out_data,
                        std::vector<Tensor<> *> &out_grad,
                        std::vector<Tensor<> *> &in_grad) override {
    CNN_UNREFERENCED_PARAMETER(in_data);
    CNN_UNREFERENCED_PARAMETER(out_data);
    for (size_t i = 0; i < num_args_; i++) *in_grad[i] = *out_grad[0];
  }

  friend struct serialization_buddy;

 private:
  size_t num_args_;
  size_t dim_;
};

}  // namespace tiny_dnn
