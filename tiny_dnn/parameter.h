/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/util/util.h"

namespace tiny_dnn {

enum parameter_type : int8_t { weight = 0x0001, bias = 0x0002 };

class parameter {
 public:
  parameter(serial_size_t width,
            serial_size_t height,
            serial_size_t depth,
            serial_size_t n_fmaps,
            param_type type,
            bool trainable = true)
    : type_(type),
      shape_(width, height, depth),
      n_fmaps_(n_fmaps),
      trainable_(trainable),
      data_(size()),
      grad_(1) {
    grad_[0].resize(data_.size());
  }

  shape3d shape() { return shape_; }

  size_t size() { return shape_.size() * n_fmaps_; }

  parameter_type type() { return param_type_; }

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

  vec_t *data() { return &data_; }

  const vec_t *data() const { return &data_; }

  void set_data(const vec_t &data) { data_ = data; }

  void set_data(const vec_t *data) { data_ = *data; }

  tensor_t *grad() { return &grad_; }

  const tensor_t *grad() const { return &grad_; }

  void set_grad(const tensor_t &grad) { grad_ = grad; }

  void set_grad(const tensor_t *grad) { grad_ = *grad; }

  void resize_grad(size_t sample_count) {
    grad_.resize(sample_count, grad_[0]);
  }

  void merge_grads(vec_t *dst) {
    assert(!grad_.empty());
    const auto &grad_head = grad_[0];
    size_t sz             = grad_head.size();
    dst->resize(sz);
    float_t *pdst = &(*dst)[0];
    // dst = grad_[0]
    std::copy(grad_head.begin(), grad_head.end(), pdst);
    // @todo consider adding parallelism
    for (size_t sample = 1, sample_count = grad_.size(); sample < sample_count;
         ++sample) {
      // dst += grad_[sample]
      vectorize::reduce<float_t>(&grad_[sample][0], sz, pdst);
    }
  }

  void clear_grads() {
    for (size_t sample = 0, sample_count = grad_.size(); sample < sample_count;
         sample++) {
      vectorize::fill(&grad_[sample][0], grad_[sample].size(), float_t{0});
    }
  }

  float_t *data_at(size_t i) { return &data_[i]; }

  float_t *grad_at(size_t sample, size_t i) { return &grad_[sample][i]; }

 private:
  parameter_type type_;
  shape3d shape_;
  size_t n_fmaps_;
  bool trainable_;

  tensor_t data_;
  tensor_t grad_;
};
}  // namespace tiny_dnn