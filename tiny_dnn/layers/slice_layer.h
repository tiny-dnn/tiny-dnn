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

enum class slice_type { slice_samples, slice_channels };

/**
 * slice an input data into multiple outputs along a given slice dimension.
 **/
class slice_layer : public layer {
 public:
  typedef layer Base;

  /**
   * @param in_shape    [in] size (width * height * channels) of input data
   * @param slice_type  [in] target axis of slicing
   * @param num_outputs [in] number of output layers
   *
   * example1:
   *   input:       NxKxWxH = 4x3x2x2  (N:batch-size, K:channels, W:width,
   *H:height)
   *   slice_type:  slice_samples
   *   num_outputs: 3
   *
   *   output[0]: 1x3x2x2
   *   output[1]: 1x3x2x2
   *   output[2]: 2x3x2x2  (mod data is assigned to the last output)
   *
   * example2:
   *   input:       NxKxWxH = 4x6x2x2
   *   slice_type:  slice_channels
   *   num_outputs: 3
   *
   *   output[0]: 4x2x2x2
   *   output[1]: 4x2x2x2
   *   output[2]: 4x2x2x2
   **/
  slice_layer(const shape3d &in_shape,
              slice_type slice_type,
              size_t num_outputs)
    : layer(std::vector<vector_type>(1, vector_type::data),
            std::vector<vector_type>(num_outputs, vector_type::data)),
      in_shape_(in_shape),
      slice_type_(slice_type),
      num_outputs_(num_outputs) {
    set_shape();
  }

  slice_layer(const layer &prev_layer,
              slice_type slice_type,
              size_t num_outputs)
    : layer(std::vector<vector_type>(1, vector_type::data),
            std::vector<vector_type>(num_outputs, vector_type::data)),
      in_shape_(prev_layer.out_shape()[0]),
      slice_type_(slice_type),
      num_outputs_(num_outputs) {
    set_shape();
  }

  std::string layer_type() const override { return "slice"; }

  std::vector<shape3d> in_shape() const override { return {in_shape_}; }

  std::vector<shape3d> out_shape() const override { return out_shapes_; }

  void forward_propagation(const std::vector<tensor_t *> &in_data,
                           std::vector<tensor_t *> &out_data) override {
    switch (slice_type_) {
      case slice_type::slice_samples:
        slice_data_forward(*in_data[0], out_data);
        break;
      case slice_type::slice_channels:
        slice_channels_forward(*in_data[0], out_data);
        break;
      default: throw nn_not_implemented_error();
    }
  }

  void back_propagation(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) override {
    CNN_UNREFERENCED_PARAMETER(in_data);
    CNN_UNREFERENCED_PARAMETER(out_data);

    switch (slice_type_) {
      case slice_type::slice_samples:
        slice_data_backward(out_grad, *in_grad[0]);
        break;
      case slice_type::slice_channels:
        slice_channels_backward(out_grad, *in_grad[0]);
        break;
      default: throw nn_not_implemented_error();
    }
  }

  slice_type get_slice_type() const { return slice_type_; }

  friend struct serialization_buddy;

 private:
  void slice_data_forward(const tensor_t &in_data,
                          std::vector<tensor_t *> &out_data) {
    const vec_t *in = &in_data[0];

    for (size_t i = 0; i < num_outputs_; i++) {
      tensor_t &out = *out_data[i];

      std::copy(in, in + slice_size_[i], &out[0]);

      in += slice_size_[i];
    }
  }

  void slice_data_backward(std::vector<tensor_t *> &out_grad,
                           tensor_t &in_grad) {
    vec_t *in = &in_grad[0];

    for (size_t i = 0; i < num_outputs_; i++) {
      tensor_t &out = *out_grad[i];

      std::copy(&out[0], &out[0] + slice_size_[i], in);

      in += slice_size_[i];
    }
  }

  void slice_channels_forward(const tensor_t &in_data,
                              std::vector<tensor_t *> &out_data) {
    size_t channel_idx       = 0;
    const size_t num_samples = in_data.size();
    const size_t spatial_dim = in_shape_.area();

    for (size_t i = 0; i < num_outputs_; i++) {
      for (size_t s = 0; s < num_samples; s++) {
        float_t *out      = &(*out_data[i])[s][0];
        const float_t *in = &in_data[s][0] + channel_idx * spatial_dim;

        std::copy(in, in + slice_size_[i] * spatial_dim, out);
      }
      channel_idx += slice_size_[i];
    }
  }

  void slice_channels_backward(std::vector<tensor_t *> &out_grad,
                               tensor_t &in_grad) {
    size_t channel_idx       = 0;
    const size_t num_samples = in_grad.size();
    const size_t spatial_dim = in_shape_.area();

    for (size_t i = 0; i < num_outputs_; i++) {
      for (size_t s = 0; s < num_samples; s++) {
        const float_t *out = &(*out_grad[i])[s][0];
        float_t *in        = &in_grad[s][0] + channel_idx * spatial_dim;

        std::copy(out, out + slice_size_[i] * spatial_dim, in);
      }
      channel_idx += slice_size_[i];
    }
  }

  void set_sample_count(size_t sample_count) override {
    if (slice_type_ == slice_type::slice_samples) {
      if (num_outputs_ == 0)
        throw nn_error("num_outputs must be positive integer");

      size_t sample_per_out = sample_count / num_outputs_;

      slice_size_.resize(num_outputs_, sample_per_out);
      slice_size_.back() = sample_count - (sample_per_out * (num_outputs_ - 1));
    }
    Base::set_sample_count(sample_count);
  }

  void set_shape() {
    switch (slice_type_) {
      case slice_type::slice_samples: set_shape_data(); break;
      case slice_type::slice_channels: set_shape_channels(); break;
      default: throw nn_not_implemented_error();
    }
  }

  void set_shape_data() { out_shapes_.resize(num_outputs_, in_shape_); }

  void set_shape_channels() {
    size_t channel_per_out = in_shape_.depth_ / num_outputs_;

    out_shapes_.resize(num_outputs_);
    for (size_t i = 0; i < num_outputs_; i++) {
      size_t ch = channel_per_out;

      if (i == num_outputs_ - 1) {
        assert(in_shape_.depth_ >= i * channel_per_out);
        ch = in_shape_.depth_ - i * channel_per_out;
      }

      slice_size_.push_back(ch);
      out_shapes_[i] = shape3d(in_shape_.width_, in_shape_.height_, ch);
    }
  }

  shape3d in_shape_;
  slice_type slice_type_;
  size_t num_outputs_;
  std::vector<shape3d> out_shapes_;
  std::vector<size_t> slice_size_;
};

}  // namespace tiny_dnn
