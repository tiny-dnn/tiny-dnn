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
 * Pad zeros around tensors.
 **/
class zero_pad_layer : public layer {
 public:
  typedef layer Base;

  /**
   * @param in_width    [in] width of input tensor
   * @param in_height   [in] height of input tensor
   * @param w_pad_size  [in] width of padding size
   * @param h_pad_size  [in] height of padding size
   */
  zero_pad_layer(size_t in_width,
            size_t in_height,
            size_t in_channels,
            size_t w_pad_size,
            size_t h_pad_size)
    : layer({vector_type::data}, {vector_type::data}),
      in_shape_(shape3d(in_width, in_height, in_channels)),
      w_pad_size_(w_pad_size),
      h_pad_size_(h_pad_size) {
    set_outshape();
  }

  /**
   * @param in_shape    [in] shape of input tensor
   * @param w_pad_size  [in] width of padding size
   * @param h_pad_size  [in] height of padding size
   */
  zero_pad_layer(const shape3d &in_shape, size_t w_pad_size, size_t h_pad_size)
    : layer({vector_type::data}, {vector_type::data}),
      in_shape_(in_shape),
      w_pad_size_(w_pad_size),
      h_pad_size_(h_pad_size) {
    set_outshape();
  }

  void set_outshape() {
    out_shape_.width_  = in_shape_.width_ + 2 * w_pad_size_;
    out_shape_.height_ = in_shape_.height_ + 2 * h_pad_size_;
    out_shape_.depth_  = in_shape_.depth_;
  }

  std::string layer_type() const override { return "zero-pad"; }

  std::vector<shape3d> in_shape() const override { return {in_shape_}; }

  std::vector<shape3d> out_shape() const override { return {out_shape_}; }

  void forward_propagation(const std::vector<tensor_t *> &in_data,
                           std::vector<tensor_t *> &out_data) override {
    const tensor_t &x = *in_data[0];
    tensor_t &y       = *out_data[0];

    for (size_t i = 0; i < x.size(); ++i) {
      for (size_t j = 0; j < x[i].size(); j++) {
        size_t col_idx        = j % in_shape_.width_;
        size_t row_idx        = j / in_shape_.width_;
        size_t depth_idx      = j / in_shape_.width_ / in_shape_.height_;
        size_t n_rows_padding = h_pad_size_ + depth_idx * 2 * h_pad_size_;
        size_t new_j          = col_idx + w_pad_size_ +
                       (row_idx + n_rows_padding) * out_shape_.width_;
        y[i][new_j] = x[i][j];
      }
    }
  }

  void back_propagation(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) override {
    tensor_t &dx       = *in_grad[0];
    const tensor_t &dy = *out_grad[0];
    const tensor_t &x  = *in_data[0];

    for (size_t i = 0; i < x.size(); i++) {
      for (size_t j = 0; j < x[i].size(); j++) {
        size_t col_idx        = j % in_shape_.width_;
        size_t row_idx        = j / in_shape_.width_;
        size_t depth_idx      = j / in_shape_.width_ / in_shape_.height_;
        size_t n_rows_padding = h_pad_size_ + depth_idx * 2 * h_pad_size_;
        size_t new_j          = col_idx + w_pad_size_ +
                       (row_idx + n_rows_padding) * out_shape_.width_;
        dx[i][j] = dy[i][new_j];
      }
    }
  }

  size_t w_pad_size() const { return w_pad_size_; }

  size_t h_pad_size() const { return h_pad_size_; }

  friend struct serialization_buddy;

 private:
  shape3d in_shape_;
  shape3d out_shape_;
  size_t w_pad_size_;
  size_t h_pad_size_;
};

}  // namespace tiny_dnn
