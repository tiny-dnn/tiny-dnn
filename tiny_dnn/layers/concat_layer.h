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
 * concat N layers along depth
 *
 * @code
 * // in: [3,1,1],[3,1,1] out: [3,1,2] (in W,H,K order)
 * concat_layer l1(2,3);
 *
 * // in: [3,2,2],[3,2,5] out: [3,2,7] (in W,H,K order)
 * concat_layer l2({shape3d(3,2,2),shape3d(3,2,5)});
 * @endcode
 **/
class concat_layer : public layer {
 public:
  /**
   * @param in_shapes [in] shapes of input tensors
   */
  explicit concat_layer(const std::vector<shape3d> &in_shapes)
    : layer(std::vector<vector_type>(in_shapes.size(), vector_type::data),
            {vector_type::data}),
      in_shapes_(in_shapes) {
    set_outshape();
  }

  /**
   * @param num_args [in] number of input tensors
   * @param ndim     [in] number of elements for each input
   */
  concat_layer(size_t num_args, size_t ndim)
    : layer(std::vector<vector_type>(num_args, vector_type::data),
            {vector_type::data}),
      in_shapes_(std::vector<shape3d>(num_args, shape3d(ndim, 1, 1))) {
    set_outshape();
  }

  void set_outshape() {
    out_shape_ = in_shapes_.front();
    for (size_t i = 1; i < in_shapes_.size(); i++) {
      if (in_shapes_[i].area() != out_shape_.area())
        throw nn_error("each input shapes to concat must have same WxH size");
      out_shape_.depth_ += in_shapes_[i].depth_;
    }
  }

  std::string layer_type() const override { return "concat"; }

  std::vector<shape3d> in_shape() const override { return in_shapes_; }

  std::vector<shape3d> out_shape() const override { return {out_shape_}; }

  void forward_propagation(const std::vector<tensor_t *> &in_data,
                           std::vector<tensor_t *> &out_data) override {
    const size_t num_samples = (*out_data[0]).size();

    for_i(num_samples, [&](size_t s) {
      float_t *outs = &(*out_data[0])[s][0];

      for (size_t i = 0; i < in_shapes_.size(); i++) {
        const float_t *ins = &(*in_data[i])[s][0];
        size_t dim         = in_shapes_[i].size();
        outs               = std::copy(ins, ins + dim, outs);
      }
    });
  }

  void back_propagation(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) override {
    CNN_UNREFERENCED_PARAMETER(in_data);
    CNN_UNREFERENCED_PARAMETER(out_data);

    size_t num_samples = (*out_grad[0]).size();

    for_i(num_samples, [&](size_t s) {
      const float_t *outs = &(*out_grad[0])[s][0];

      for (size_t i = 0; i < in_shapes_.size(); i++) {
        size_t dim   = in_shapes_[i].size();
        float_t *ins = &(*in_grad[i])[s][0];
        std::copy(outs, outs + dim, ins);
        outs += dim;
      }
    });
  }

  friend struct serialization_buddy;

 private:
  std::vector<shape3d> in_shapes_;
  shape3d out_shape_;
};

}  // namespace tiny_dnn
