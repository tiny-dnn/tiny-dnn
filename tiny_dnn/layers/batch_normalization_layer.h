/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

#include "tiny_dnn/layers/layer.h"
#include "tiny_dnn/util/math_functions.h"
#include "tiny_dnn/util/util.h"

namespace tiny_dnn {

/**
 * Batch Normalization
 *
 * Normalize the activations of the previous layer at each batch
 **/
class batch_normalization_layer : public layer {
 public:
  typedef layer Base;

  /**
   * @param prev_layer      [in] previous layer to be connected with this layer
   * @param epsilon         [in] small positive value to avoid zero-division
   * @param momentum        [in] momentum in the computation of the exponential
   *average of the mean/stddev of the data
   * @param phase           [in] specify the current context (train/test)
   **/
  batch_normalization_layer(const layer &prev_layer,
                            float_t epsilon  = 1e-5,
                            float_t momentum = 0.999,
                            net_phase phase  = net_phase::train)
    : Base({vector_type::data}, {vector_type::data}),
      in_channels_(prev_layer.out_shape()[0].depth_),
      in_spatial_size_(prev_layer.out_shape()[0].area()),
      phase_(phase),
      momentum_(momentum),
      eps_(epsilon),
      update_immidiately_(false) {
    init();
  }

  /**
   * @param in_spatial_size [in] spatial size (WxH) of the input data
   * @param in_channels     [in] channels of the input data
   * @param epsilon         [in] small positive value to avoid zero-division
   * @param momentum        [in] momentum in the computation of the exponential
   *average of the mean/stddev of the data
   * @param phase           [in] specify the current context (train/test)
   **/
  batch_normalization_layer(size_t in_spatial_size,
                            size_t in_channels,
                            float_t epsilon  = 1e-5,
                            float_t momentum = 0.999,
                            net_phase phase  = net_phase::train)
    : Base({vector_type::data}, {vector_type::data}),
      in_channels_(in_channels),
      in_spatial_size_(in_spatial_size),
      phase_(phase),
      momentum_(momentum),
      eps_(epsilon),
      update_immidiately_(false) {
    init();
  }

  virtual ~batch_normalization_layer() {}

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

  void back_propagation(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) override {
    tensor_t &prev_delta     = *in_grad[0];
    tensor_t &curr_delta     = *out_grad[0];
    const tensor_t &curr_out = *out_data[0];
    const size_t num_samples = curr_out.size();

    CNN_UNREFERENCED_PARAMETER(in_data);

    tensor_t delta_dot_y = curr_out;
    vec_t mean_delta_dot_y, mean_delta, mean_Y;

    for (size_t i = 0; i < num_samples; i++) {
      for (size_t j = 0; j < curr_out[0].size(); j++) {
        delta_dot_y[i][j] *= curr_delta[i][j];
      }
    }

    moments(delta_dot_y, in_spatial_size_, in_channels_, mean_delta_dot_y);
    moments(curr_delta, in_spatial_size_, in_channels_, mean_delta);
    // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
    //
    // dE(Y)/dX =
    //   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
    //     ./ sqrt(var(X) + eps)
    //
    for_i(num_samples, [&](size_t i) {
      for (size_t j = 0; j < in_channels_; j++) {
        for (size_t k = 0; k < in_spatial_size_; k++) {
          size_t index = j * in_spatial_size_ + k;

          prev_delta[i][index] = curr_delta[i][index] - mean_delta[j] -
                                 mean_delta_dot_y[j] * curr_out[i][index];

          // stddev_ is calculated in the forward pass
          prev_delta[i][index] /= stddev_[j];
        }
      }
    });
  }

  void forward_propagation(const std::vector<tensor_t *> &in_data,
                           std::vector<tensor_t *> &out_data) override {
    vec_t &mean = (phase_ == net_phase::train) ? mean_current_ : mean_;
    vec_t &variance =
      (phase_ == net_phase::train) ? variance_current_ : variance_;
    tensor_t &in  = *in_data[0];
    tensor_t &out = *out_data[0];

    if (phase_ == net_phase::train) {
      // calculate mean/variance from this batch in train phase
      moments(*in_data[0], in_spatial_size_, in_channels_, mean, variance);
    }

    // y = (x - mean) ./ sqrt(variance + eps)
    calc_stddev(variance);

    for_i(in_data[0]->size(), [&](size_t i) {
      const float_t *inptr = &in[i][0];
      float_t *outptr      = &out[i][0];

      for (size_t j = 0; j < in_channels_; j++) {
        float_t m = mean[j];

        for (size_t k = 0; k < in_spatial_size_; k++) {
          *outptr++ = (*inptr++ - m) / stddev_[j];
        }
      }
    });

    if (phase_ == net_phase::train && update_immidiately_) {
      mean_     = mean_current_;
      variance_ = variance_current_;
    }
  }

  void set_context(net_phase ctx) override { phase_ = ctx; }

  std::string layer_type() const override { return "batch-norm"; }

  void post_update() override {
    for (size_t i = 0; i < mean_.size(); i++) {
      mean_[i] = momentum_ * mean_[i] + (1 - momentum_) * mean_current_[i];
      variance_[i] =
        momentum_ * variance_[i] + (1 - momentum_) * variance_current_[i];
    }
  }

  void save(
    std::ostream &os,
    const int precision = std::numeric_limits<float_t>::digits10 + 2
    /*by default, we want there to be enough precision*/) const override {
    Base::save(os, precision);
    for (auto m : mean_) os << m << " ";
    for (auto v : variance_) os << v << " ";
  }

  void load(std::istream &is,
            const int precision = std::numeric_limits<float_t>::digits10 + 2
            /*by default, we want there to be enough precision*/) override {
    Base::load(is, precision);
    for (auto &m : mean_) is >> m;
    for (auto &v : variance_) is >> v;
  }

  void load(const std::vector<float_t> &src, int &idx) override {
    Base::load(src, idx);
    for (auto &m : mean_) m     = src[idx++];
    for (auto &v : variance_) v = src[idx++];
  }

  void update_immidiately(bool update) { update_immidiately_ = update; }

  void set_stddev(const vec_t &stddev) { stddev_ = stddev; }

  void set_mean(const vec_t &mean) { mean_ = mean; }

  void set_variance(const vec_t &variance) {
    variance_ = variance;
    calc_stddev(variance);
  }

  float_t epsilon() const { return eps_; }

  float_t momentum() const { return momentum_; }

  friend struct serialization_buddy;

 private:
  void calc_stddev(const vec_t &variance) {
    for (size_t i = 0; i < in_channels_; i++) {
      stddev_[i] = sqrt(variance[i] + eps_);
    }
  }

  void init() {
    mean_current_.resize(in_channels_);
    mean_.resize(in_channels_);
    variance_current_.resize(in_channels_);
    variance_.resize(in_channels_);
    tmp_mean_.resize(in_channels_);
    stddev_.resize(in_channels_);
  }

  size_t in_channels_;
  size_t in_spatial_size_;

  net_phase phase_;
  float_t momentum_;
  float_t eps_;

  // mean/variance for this mini-batch
  vec_t mean_current_;
  vec_t variance_current_;

  vec_t tmp_mean_;

  // moving average of mean/variance
  vec_t mean_;
  vec_t variance_;
  vec_t stddev_;

  // for test
  bool update_immidiately_;
};

}  // namespace tiny_dnn
