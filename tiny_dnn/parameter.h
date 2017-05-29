/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/util/util.h"

namespace tiny_dnn {
class parameter : public std::enable_shared_from_this<parameter> {
 public:
  enum class param_type : int8_t { weight = 0x0001, bias = 0x0002 };

  parameter(serial_size_t width,
            serial_size_t height,
            serial_size_t depth,
            serial_size_t n_fmaps,
            param_type type,
            bool trainable = true)
    : param_type_(type),
      shape_(width, height, depth),
      n_fmaps_(n_fmaps),
      trainable_(trainable),
      data_(size()),
      grad_(size()) {}

  shape3d get_shape() { return shape_; }

  size_t size() { return shape_.size() * n_fmaps_; }

  param_type get_param_type() { return param_type_; }

  void set_width(serial_size_t width) { shape_.width_ = width; }

  void set_height(serial_size_t height) { shape_.height_ = height; }

  void set_depth(serial_size_t depth) { shape_.depth_ = depth; }

  void set_fmaps(serial_size_t n_fmaps) { n_fmaps_ = n_fmaps; }

  void set_dims(serial_size_t width,
                serial_size_t height,
                serial_size_t depth,
                serial_size_t n_fmaps) {
    shape_.width_  = width;
    shape_.height_ = height;
    shape_.depth_  = depth;
    n_fmaps_       = n_fmaps;
  }

  bool is_trainable() { return trainable_; }

  void set_trainable() { trainable_ = true; }

  void freeze_trainable() { trainable_ = false; }

  vec_t *get_data() { return &data_; }

  const vec_t *get_data() const { return &data_; }

  void set_data(const vec_t &data) { data_ = data; }

  void set_data(const vec_t *data) { data_ = *data; }

  vec_t *get_grad() { return &grad_; }

  const vec_t *get_grad() const { return &grad_; }

  void set_grad(const vec_t &grad) { grad_ = grad; }

  void set_grad(const vec_t *grad) { grad_ = *grad; }

  void merge_grads(vec_t *dst) {
    dst->resize(grad_.size());
    float_t *pdst = &(*dst)[0];
    std::copy(grad_.begin(), grad_.end(), pdst);
    vectorize::reduce<float_t>(&grad_[0], grad_.size(), pdst);
  }

  void clear_grads() { vectorize::fill(&grad_[0], grad_.size(), float_t{0}); }

  float_t *data_at(size_t i) { return &data_[i]; }

  float_t *grad_at(size_t i) { return &grad_[i]; }

 private:
  param_type param_type_;
  shape3d shape_;
  size_t n_fmaps_;
  bool trainable_;

  vec_t data_;
  vec_t grad_;
};
}  // namespace tiny_dnn
