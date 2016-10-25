// Copyright (c) 2013-2016, Taiga Nomi. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#pragma once
#include "tiny_dnn/layers/layer.h"
#include "tiny_dnn/util/util.h"

namespace tiny_dnn {

/**
 * concat N layers along depth
 **/
class concat_layer : public layer {
 public:
  concat_layer(const std::vector<shape3d>& in_shapes)
      : layer(std::vector<vector_type>(in_shapes.size(), vector_type::data),
              {vector_type::data}),
        in_shapes_(in_shapes) {
    set_outshape();
  }

  concat_layer(cnn_size_t num_args, cnn_size_t ndim)
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

  void forward_propagation(const std::vector<tensor_t*>& in_data,
                           std::vector<tensor_t*>& out_data) override {
    cnn_size_t num_samples = (*out_data[0]).size();

    for (cnn_size_t s = 0; s < num_samples; s++) {
      float_t* outs = &(*out_data[0])[s][0];

      for (cnn_size_t i = 0; i < in_shapes_.size(); i++) {
        const float_t* ins = &(*in_data[i])[s][0];
        cnn_size_t dim = in_shapes_[i].size();
        outs = std::copy(ins, ins + dim, outs);
      }
    }
  }

  void back_propagation(const std::vector<tensor_t*>& in_data,
                        const std::vector<tensor_t*>& out_data,
                        std::vector<tensor_t*>& out_grad,
                        std::vector<tensor_t*>& in_grad) override {
    CNN_UNREFERENCED_PARAMETER(in_data);
    CNN_UNREFERENCED_PARAMETER(out_data);

    cnn_size_t num_samples = (*out_grad[0]).size();

    for (cnn_size_t s = 0; s < num_samples; s++) {
      const float_t* outs = &(*out_grad[0])[s][0];

      for (cnn_size_t i = 0; i < in_shapes_.size(); i++) {
        cnn_size_t dim = in_shapes_[i].size();
        float_t* ins = &(*in_grad[i])[s][0];
        std::copy(outs, outs + dim, ins);
        outs += dim;
      }
    }
  }

  template <class Archive>
  static void load_and_construct(Archive& ar,
                                 cereal::construct<concat_layer>& construct) {
    std::vector<shape3d> in_shapes;

    ar(in_shapes);
    construct(in_shapes);
  }

  template <class Archive>
  void serialize(Archive& ar) {
    layer::serialize_prolog(ar);
    ar(in_shapes_);
  }

 private:
  std::vector<shape3d> in_shapes_;
  shape3d out_shape_;
};

}  // namespace tiny_dnn
