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

#include "tiny_dnn/util/util.h"

namespace tiny_dnn {

enum class norm_region { across_channels, within_channels };

/**
 * local response normalization
 */
class lrn_layer : public layer {
 public:
  lrn_layer(const shape3d &in_shape,
            size_t local_size,
            float_t alpha      = 1.0,
            float_t beta       = 5.0,
            norm_region region = norm_region::across_channels)
    : layer({vector_type::data}, {vector_type::data}),
      in_shape_(in_shape),
      size_(local_size),
      alpha_(alpha),
      beta_(beta),
      region_(region),
      in_square_(in_shape_.area()) {}

  /**
   * @param layer       [in] the previous layer connected to this
   * @param local_size  [in] the number of channels(depths) to sum over
   * @param in_channels [in] the number of channels of input data
   * @param alpha       [in] the scaling parameter (same to caffe's LRN)
   * @param beta        [in] the scaling parameter (same to caffe's LRN)
   **/
  lrn_layer(layer *prev,
            size_t local_size,
            float_t alpha      = 1.0,
            float_t beta       = 5.0,
            norm_region region = norm_region::across_channels)
    : lrn_layer(prev->out_data_shape()[0], local_size, alpha, beta, region) {}

  /**
   * @param in_width    [in] the width of input data
   * @param in_height   [in] the height of input data
   * @param local_size  [in] the number of channels(depths) to sum over
   * @param in_channels [in] the number of channels of input data
   * @param alpha       [in] the scaling parameter (same to caffe's LRN)
   * @param beta        [in] the scaling parameter (same to caffe's LRN)
   **/
  lrn_layer(size_t in_width,
            size_t in_height,
            size_t local_size,
            size_t in_channels,
            float_t alpha      = 1.0,
            float_t beta       = 5.0,
            norm_region region = norm_region::across_channels)
    : lrn_layer(shape3d{in_width, in_height, in_channels},
                local_size,
                alpha,
                beta,
                region) {}

  size_t fan_in_size() const override { return size_; }

  size_t fan_out_size() const override { return size_; }

  std::vector<shape3d> in_shape() const override { return {in_shape_}; }

  std::vector<shape3d> out_shape() const override { return {in_shape_}; }

  std::string layer_type() const override { return "lrn"; }

  void forward_propagation(const std::vector<tensor_t *> &in_data,
                           std::vector<tensor_t *> &out_data) override {
    // @todo revise the parallelism strategy
    for (size_t sample = 0, sample_count = in_data[0]->size();
         sample < sample_count; ++sample) {
      vec_t &in  = (*in_data[0])[sample];
      vec_t &out = (*out_data[0])[sample];

      if (region_ == norm_region::across_channels) {
        forward_across(in, out);
      } else {
        forward_within(in, out);
      }
    }
  }

  void back_propagation(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) override {
    CNN_UNREFERENCED_PARAMETER(in_data);
    CNN_UNREFERENCED_PARAMETER(out_data);
    CNN_UNREFERENCED_PARAMETER(out_grad);
    CNN_UNREFERENCED_PARAMETER(in_grad);
    throw nn_error("not implemented");
  }

  friend struct serialization_buddy;

 private:
  void forward_across(const vec_t &in, vec_t &out) {
    vectorize::fill(&in_square_[0], in_square_.size(), float_t{0});

    for (size_t i = 0; i < size_ / 2; i++) {
      size_t idx = in_shape_.get_index(0, 0, i);
      add_square_sum(&in[idx], in_shape_.area(), &in_square_[0]);
    }

    size_t head = size_ / 2;
    long tail   = static_cast<long>(head) - static_cast<long>(size_);  // NOLINT
    size_t channels              = in_shape_.depth_;
    const size_t wxh             = in_shape_.area();
    const float_t alpha_div_size = alpha_ / size_;

    for (size_t i = 0; i < channels; i++, head++, tail++) {
      if (head < channels)
        add_square_sum(&in[in_shape_.get_index(0, 0, head)], wxh,
                       &in_square_[0]);

      if (tail >= 0)
        sub_square_sum(&in[in_shape_.get_index(0, 0, tail)], wxh,
                       &in_square_[0]);

      float_t *dst       = &out[in_shape_.get_index(0, 0, i)];
      const float_t *src = &in[in_shape_.get_index(0, 0, i)];
      for (size_t j = 0; j < wxh; j++)
        dst[j]      = src[j] *
                 std::pow(float_t(1) + alpha_div_size * in_square_[j], -beta_);
    }
  }

  void forward_within(const vec_t &in, vec_t &out) {
    CNN_UNREFERENCED_PARAMETER(in);
    CNN_UNREFERENCED_PARAMETER(out);
    throw nn_error("not implemented");
  }

  void add_square_sum(const float_t *src, size_t size, float_t *dst) {
    for (size_t i = 0; i < size; i++) dst[i] += src[i] * src[i];
  }

  void sub_square_sum(const float_t *src, size_t size, float_t *dst) {
    for (size_t i = 0; i < size; i++) dst[i] -= src[i] * src[i];
  }

  shape3d in_shape_;

  size_t size_;
  float_t alpha_, beta_;
  norm_region region_;

  vec_t in_square_;
};

}  // namespace tiny_dnn
