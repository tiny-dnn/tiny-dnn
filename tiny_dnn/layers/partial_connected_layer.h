/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <utility>
#include <vector>

#include "tiny_dnn/layers/layer.h"
#include "tiny_dnn/util/util.h"

namespace tiny_dnn {

class partial_connected_layer : public layer {
 public:
  typedef std::vector<std::pair<size_t, size_t>> io_connections;
  typedef std::vector<std::pair<size_t, size_t>> wi_connections;
  typedef std::vector<std::pair<size_t, size_t>> wo_connections;

  partial_connected_layer(size_t in_dim,
                          size_t out_dim,
                          size_t weight_dim,
                          size_t bias_dim,
                          float_t scale_factor = float_t{1})
    : layer(std_input_order(bias_dim > 0), {vector_type::data}),
      weight2io_(weight_dim),
      out2wi_(out_dim),
      in2wo_(in_dim),
      bias2out_(bias_dim),
      out2bias_(out_dim),
      scale_factor_(scale_factor) {}

  size_t param_size() const {
    size_t total_param = 0;
    for (auto w : weight2io_)
      if (w.size() > 0) total_param++;
    for (auto b : bias2out_)
      if (b.size() > 0) total_param++;
    return total_param;
  }

  size_t fan_in_size() const override { return max_size(out2wi_); }

  size_t fan_out_size() const override { return max_size(in2wo_); }

  void connect_weight(size_t input_index,
                      size_t output_index,
                      size_t weight_index) {
    weight2io_[weight_index].emplace_back(input_index, output_index);
    out2wi_[output_index].emplace_back(weight_index, input_index);
    in2wo_[input_index].emplace_back(weight_index, output_index);
  }

  void connect_bias(size_t bias_index, size_t output_index) {
    out2bias_[output_index] = bias_index;
    bias2out_[bias_index].push_back(output_index);
  }

  void forward_propagation(const std::vector<tensor_t *> &in_data,
                           std::vector<tensor_t *> &out_data) override {
    const tensor_t &in = *in_data[0];
    const vec_t &W     = (*in_data[1])[0];
    const vec_t &b     = (*in_data[2])[0];
    tensor_t &out      = *out_data[0];

    // @todo revise the parallelism strategy
    for (size_t sample = 0, sample_count = in.size(); sample < sample_count;
         ++sample) {
      vec_t &out_sample = out[sample];

      for_i(out2wi_.size(), [&](size_t i) {
        const wi_connections &connections = out2wi_[i];

        float_t &out_element = out_sample[i];

        out_element = float_t{0};

        for (auto connection : connections)
          out_element += W[connection.first] * in[sample][connection.second];

        out_element *= scale_factor_;
        out_element += b[out2bias_[i]];
      });
    }
  }

  void back_propagation(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) override {
    CNN_UNREFERENCED_PARAMETER(out_data);
    const tensor_t &prev_out = *in_data[0];
    const vec_t &W           = (*in_data[1])[0];
    vec_t &dW                = (*in_grad[1])[0];
    vec_t &db                = (*in_grad[2])[0];
    tensor_t &prev_delta     = *in_grad[0];
    tensor_t &curr_delta     = *out_grad[0];

    // @todo revise the parallelism strategy
    for (size_t sample = 0, sample_count = prev_out.size();
         sample < sample_count; ++sample) {
      for_i(in2wo_.size(), [&](size_t i) {
        const wo_connections &connections = in2wo_[i];
        float_t delta{0};

        for (auto connection : connections)
          delta += W[connection.first] * curr_delta[sample][connection.second];

        prev_delta[sample][i] = delta * scale_factor_;
      });

      for_i(weight2io_.size(), [&](size_t i) {
        const io_connections &connections = weight2io_[i];
        float_t diff{0};

        for (auto connection : connections)
          diff += prev_out[sample][connection.first] *
                  curr_delta[sample][connection.second];

        dW[i] += diff * scale_factor_;
      });

      for (size_t i = 0; i < bias2out_.size(); i++) {
        const std::vector<size_t> &outs = bias2out_[i];
        float_t diff{0};

        for (auto o : outs) diff += curr_delta[sample][o];

        db[i] += diff;
      }
    }
  }

  friend struct serialization_buddy;

 protected:
  std::vector<io_connections> weight2io_;  // weight_id -> [(in_id, out_id)]
  std::vector<wi_connections> out2wi_;     // out_id -> [(weight_id, in_id)]
  std::vector<wo_connections> in2wo_;      // in_id -> [(weight_id, out_id)]
  std::vector<std::vector<size_t>> bias2out_;
  std::vector<size_t> out2bias_;
  float_t scale_factor_;
};

}  // namespace tiny_dnn
