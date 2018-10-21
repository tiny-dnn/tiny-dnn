/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "tiny_dnn/layers/layer.h"
#include "tiny_dnn/util/math_functions.h"
#include "tiny_dnn/util/util.h"

namespace tiny_dnn {

/**
 * L2 Normalization
 *
 * Normalize the activations of the previous layer at each batch
 **/
class l2_normalization_layer : public layer {
 public:
  typedef layer Base;

  /**
   * @param prev_layer      [in] previous layer to be connected with this layer
   * @param epsilon         [in] small positive value to avoid zero-division
   * @param scale           [in] scale factor to multiply after normalization
   **/
  l2_normalization_layer(const layer &prev_layer,
                         float_t epsilon = 1e-10,
                         float_t scale   = 20)
    : Base({vector_type::data}, {vector_type::data}),
      in_channels_(prev_layer.out_shape()[0].depth_),
      in_spatial_size_(prev_layer.out_shape()[0].area()),
      eps_(std::max(epsilon, std::numeric_limits<float_t>::epsilon())),
      scale_(scale) {}

  /**
   * @param in_spatial_size [in] spatial size (WxH) of the input data
   * @param in_channels     [in] channels of the input data
   * @param epsilon         [in] small positive value to avoid zero-division
   * @param scale           [in] scale factor to multiply after normalization
   **/
  l2_normalization_layer(size_t in_spatial_size,
                         size_t in_channels,
                         float_t epsilon = 1e-10,
                         float_t scale   = 20)
    : Base({vector_type::data}, {vector_type::data}),
      in_channels_(in_channels),
      in_spatial_size_(in_spatial_size),
      eps_(std::max(epsilon, std::numeric_limits<float_t>::epsilon())),
      scale_(scale) {}

  virtual ~l2_normalization_layer() {}

  ///< number of incoming connections for each output unit
  size_t fan_in_size() const override { return 1; }

  ///< number of outgoing connections for each input unit
  size_t fan_out_size() const override { return 1; }

  std::vector<index3d<size_t>> in_shape() const override {
    return {index3d<size_t>(in_spatial_size_, 1, in_channels_)};
  }

  std::vector<index3d<size_t>> out_shape() const override {
    return {index3d<size_t>(in_spatial_size_, 1, in_channels_)};
  }

  void forward_propagation(const std::vector<tensor_t *> &in_data,
                           std::vector<tensor_t *> &out_data) override {
    tensor_t &in  = *in_data[0];
    tensor_t &out = *out_data[0];

    for_i(in_data[0]->size(), [&](size_t i) {
      const float_t *inptr = &in[i][0];
      float_t *outptr      = &out[i][0];

      for (size_t j = 0; j < in_spatial_size_; ++j) {
        float_t sum_of_square = 0;
        for (size_t k = 0; k < in_channels_; ++k) {
          float_t value = *(inptr + k * in_spatial_size_);
          sum_of_square += value * value;
        }
        sum_of_square            = std::max(sum_of_square, eps_);
        float_t root_sum_squared = sqrt(sum_of_square);

        for (size_t k = 0; k < in_channels_; ++k) {
          const float_t *inptr_c = inptr + k * in_spatial_size_;
          float_t *outptr_c      = outptr + k * in_spatial_size_;
          *outptr_c              = *inptr_c / root_sum_squared * scale_;
        }
        ++inptr;
        ++outptr;
      }
    });
  }

  void back_propagation(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) override {
    tensor_t &prev_delta     = *in_grad[0];
    tensor_t &curr_delta     = *out_grad[0];
    const tensor_t &curr_out = *out_data[0];
    const size_t num_samples = curr_out.size();

    CNN_UNREFERENCED_PARAMETER(in_data);
  }

  std::string layer_type() const override { return "l2-norm"; }

  float_t epsilon() const { return eps_; }

  float_t scale() const { return scale_; }

  friend struct serialization_buddy;

 private:
  size_t in_channels_;
  size_t in_spatial_size_;

  float_t eps_;
  float_t scale_;
};

}  // namespace tiny_dnn
