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
 * applies dropout to the input
 **/
class dropout_layer : public layer {
 public:
  /**
   * @param in_dim       [in] number of elements of the input
   * @param dropout_rate [in] (0-1) fraction of the input units to be dropped
   * @param phase        [in] initial state of the dropout
   **/
  dropout_layer(size_t in_dim,
                float_t dropout_rate,
                net_phase phase = net_phase::train)
    : layer({vector_type::data}, {vector_type::data}),
      phase_(phase),
      dropout_rate_(dropout_rate),
      scale_(float_t(1) / (float_t(1) - dropout_rate_)),
      in_size_(in_dim) {
    mask_.resize(1, std::vector<uint8_t>(in_dim));
    clear_mask();
  }

  dropout_layer(const dropout_layer &obj) = default;
  virtual ~dropout_layer() {}

  dropout_layer(dropout_layer &&obj) = default;
  dropout_layer &operator=(const dropout_layer &obj) = default;
  dropout_layer &operator=(dropout_layer &&obj) = default;

  void set_dropout_rate(float_t rate) {
    dropout_rate_ = rate;
    scale_        = float_t(1) / (float_t(1) - dropout_rate_);
  }

  float_t dropout_rate() const { return dropout_rate_; }

  ///< number of incoming connections for each output unit
  size_t fan_in_size() const override { return 1; }

  ///< number of outgoing connections for each input unit
  size_t fan_out_size() const override { return 1; }

  std::vector<index3d<size_t>> in_shape() const override {
    return {index3d<size_t>(in_size_, 1, 1)};
  }

  std::vector<index3d<size_t>> out_shape() const override {
    return {index3d<size_t>(in_size_, 1, 1)};
  }

  void back_propagation(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) override {
    tensor_t &prev_delta       = *in_grad[0];
    const tensor_t &curr_delta = *out_grad[0];

    CNN_UNREFERENCED_PARAMETER(in_data);
    CNN_UNREFERENCED_PARAMETER(out_data);

    for_i(prev_delta.size(), [&](size_t sample) {
      // assert(prev_delta[sample].size() == curr_delta[sample].size());
      // assert(mask_[sample].size() == prev_delta[sample].size());
      size_t sz = prev_delta[sample].size();
      for (size_t i = 0; i < sz; ++i) {
        prev_delta[sample][i] = mask_[sample][i] * curr_delta[sample][i];
      }
    });
  }

  void forward_propagation(const std::vector<tensor_t *> &in_data,
                           std::vector<tensor_t *> &out_data) override {
    const tensor_t &in = *in_data[0];
    tensor_t &out      = *out_data[0];

    const size_t sample_count = in.size();

    if (mask_.size() < sample_count) {
      mask_.resize(sample_count, mask_[0]);
    }

    for_i(sample_count, [&](size_t sample) {
      std::vector<uint8_t> &mask = mask_[sample];

      const vec_t &in_vec = in[sample];
      vec_t &out_vec      = out[sample];

      if (phase_ == net_phase::train) {
        for (size_t i = 0; i < in_vec.size(); i++)
          mask[i]     = bernoulli(dropout_rate_);

        for (size_t i = 0; i < in_vec.size(); i++)
          out_vec[i]  = mask[i] * scale_ * in_vec[i];
      } else {
        for (size_t i = 0, end = in_vec.size(); i < end; i++)
          out_vec[i] = in_vec[i];
      }
    });
  }

  /**
   * set dropout-context (training-phase or test-phase)
   **/
  void set_context(net_phase ctx) override { phase_ = ctx; }

  std::string layer_type() const override { return "dropout"; }

  // currently used by tests only
  const std::vector<uint8_t> &get_mask(size_t sample_index) const {
    return mask_[sample_index];
  }

  void clear_mask() {
    for (auto &sample : mask_) {
      std::fill(sample.begin(), sample.end(), 0);
    }
  }

  friend struct serialization_buddy;

 private:
  net_phase phase_;
  float_t dropout_rate_;
  float_t scale_;
  size_t in_size_;
  std::vector<std::vector<uint8_t>> mask_;
};

}  // namespace tiny_dnn
