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

#include "tiny_dnn/core/params/aveunpool_params.h"
#include "tiny_dnn/util/util.h"

#ifdef DNN_USE_IMAGE_API
#include "tiny_dnn/util/image.h"
#endif  // DNN_USE_IMAGE_API

namespace tiny_dnn {

// forward_propagation
inline void tiny_average_unpooling_kernel(
  const std::vector<tensor_t *> &in_data,
  std::vector<tensor_t *> &out_data,
  const core::aveunpool_params &params,
  const core::average_unpooling_layer_worker_specific_storage &auws,
  bool parallelize) {
  for_i(parallelize, in_data[0]->size(), [&](size_t sample) {
    const Tensor<> in((*in_data[0])[sample]);
    const Tensor<> W((*in_data[1])[0]);
    const Tensor<> b((*in_data[2])[0]);
    Tensor<> out((*out_data[0])[sample]);

    auto oarea = params.out.area();
    size_t idx = 0;
    for (size_t d = 0; d < params.out.depth_; ++d) {
      float_t weight = W.host_at(d);  // * scale_factor;
      float_t bias   = b.host_at(d);
      for (size_t i = 0; i < oarea; ++i, ++idx) {
        const auto &connections = auws.out2wi[idx];
        float_t value{0};
        for (auto connection : connections)
          value += in.host_at(connection.second);
        value *= weight;
        value += bias;
        out.host_at(idx) = value;
      }
    }

    assert(out.size() == auws.out2wi.size());
    (*out_data[0])[sample] = out.toVec();
  });
}

// back_propagation
inline void tiny_average_unpooling_back_kernel(
  const std::vector<tensor_t *> &in_data,
  const std::vector<tensor_t *> &out_data,
  std::vector<tensor_t *> &out_grad,
  std::vector<tensor_t *> &in_grad,
  const core::aveunpool_params &params,
  const core::average_unpooling_layer_worker_specific_storage &auws,
  bool parallelize) {
  for_i(parallelize, in_data[0]->size(), [&](size_t sample) {
    const Tensor<> prev_out((*in_data[0])[sample]);
    const Tensor<> W((*in_data[1])[0]);
    Tensor<> dW((*in_grad[1])[sample]);
    Tensor<> db((*in_grad[2])[sample]);
    Tensor<> prev_delta((*in_grad[0])[sample]);
    Tensor<> curr_delta((*out_grad[0])[sample]);

    auto inarea = params.in.area();
    size_t idx  = 0;
    for (size_t i = 0; i < params.in.depth_; ++i) {
      float_t weight = W.host_at(i);  // * scale_factor;
      for (size_t j = 0; j < inarea; ++j, ++idx) {
        prev_delta.host_at(idx) =
          weight * curr_delta.host_at(auws.in2wo[idx][0].second);
      }
    }

    for (size_t i = 0; i < auws.weight2io.size(); ++i) {
      const auto &connections = auws.weight2io[i];
      float_t diff            = 0.0;

      for (auto connection : connections)
        diff += prev_out.host_at(connection.first) *
                curr_delta.host_at(connection.second);

      dW.host_at(i) += diff;  // * scale_factor;
    }

    for (size_t i = 0; i < auws.bias2out.size(); i++) {
      const std::vector<size_t> &outs = auws.bias2out[i];
      float_t diff                    = 0.0;

      for (auto o : outs) diff += curr_delta.host_at(o);

      db.host_at(i) += diff;
    }
    (*in_grad[1])[sample]  = dW.toVec();
    (*in_grad[2])[sample]  = db.toVec();
    (*in_grad[0])[sample]  = prev_delta.toVec();
    (*out_grad[0])[sample] = curr_delta.toVec();
  });
}

/**
 * average pooling with trainable weights
 **/
class average_unpooling_layer : public layer {
 public:
  /**
   * @param in_width     [in] width of input image
   * @param in_height    [in] height of input image
   * @param in_channels  [in] the number of input image channels(depth)
   * @param pooling_size [in] factor by which to upscale
   **/
  average_unpooling_layer(size_t in_width,
                          size_t in_height,
                          size_t in_channels,
                          size_t pooling_size)
    : average_unpooling_layer(
        in_width, in_height, in_channels, pooling_size, pooling_size) {}

  /**
   * @param in_width     [in] width of input image
   * @param in_height    [in] height of input image
   * @param in_channels  [in] the number of input image channels(depth)
   * @param pooling_size [in] factor by which to upscale
   * @param stride       [in] interval at which to apply the filters to the
   *input
   **/
  average_unpooling_layer(size_t in_width,
                          size_t in_height,
                          size_t in_channels,
                          size_t pooling_size,
                          size_t stride)
    : layer({vector_type::data, vector_type::weight, vector_type::bias},
            {vector_type::data}) {
    aveunpool_set_params(in_width, in_height, in_channels, pooling_size,
                         stride);
    auws_ = core::average_unpooling_layer_worker_specific_storage(
      in_channels, params_.out.size(), params_.in.size(), in_channels);
    init_connection(pooling_size);
  }

  std::vector<shape3d> in_shape() const override {
    return {params_.in, params_.window, params_.out};
  }

  std::vector<shape3d> out_shape() const override { return {params_.out}; }

  std::string layer_type() const override { return "ave-unpool"; }

  void forward_propagation(const std::vector<tensor_t *> &in_data,
                           std::vector<tensor_t *> &out_data) override {
    tiny_average_unpooling_kernel(in_data, out_data, params_, auws_,
                                  parallelize());
  }

  void back_propagation(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) override {
    tiny_average_unpooling_back_kernel(in_data, out_data, out_grad, in_grad,
                                       params_, auws_, parallelize());
  }

  friend struct serialization_buddy;

 private:
  core::aveunpool_params params_;
  core::average_unpooling_layer_worker_specific_storage auws_;

  void aveunpool_set_params(size_t in_width,
                            size_t in_height,
                            size_t in_channels,
                            size_t pooling_size,
                            size_t stride) {
    params_.in  = shape3d(in_width, in_height, in_channels);
    params_.out = shape3d((in_width - 1) * stride + pooling_size,
                          (in_height - 1) * stride + pooling_size, in_channels);
    params_.window       = shape3d(pooling_size, pooling_size, in_channels);
    params_.stride       = stride;
    params_.scale_factor = sqr(pooling_size);
  }

  void init_connection(size_t pooling_size) {
    for (size_t c = 0; c < params_.in.depth_; ++c) {
      for (size_t y = 0; y < params_.in.height_; ++y) {
        for (size_t x = 0; x < params_.in.width_; ++x) {
          connect_kernel(pooling_size, x, y, c);
        }
      }
    }

    for (size_t c = 0; c < params_.in.depth_; ++c) {
      for (size_t y = 0; y < params_.out.height_; ++y) {
        for (size_t x = 0; x < params_.out.width_; ++x) {
          auws_.bias2out[c].push_back(params_.out.get_index(x, y, c));
        }
      }
    }
  }

  void connect_kernel(size_t pooling_size, size_t x, size_t y, size_t inc) {
    size_t dymax = std::min(pooling_size, params_.out.height_ - y);
    size_t dxmax = std::min(pooling_size, params_.out.width_ - x);
    size_t dstx  = x * params_.stride;
    size_t dsty  = y * params_.stride;
    size_t inidx = params_.in.get_index(x, y, inc);
    for (size_t dy = 0; dy < dymax; ++dy) {
      for (size_t dx = 0; dx < dxmax; ++dx) {
        connect_weight(inidx, params_.out.get_index(dstx + dx, dsty + dy, inc),
                       inc);
      }
    }
  }

  void connect_weight(size_t input_index,
                      size_t output_index,
                      size_t weight_index) {
    auws_.weight2io[weight_index].emplace_back(input_index, output_index);
    auws_.out2wi[output_index].emplace_back(weight_index, input_index);
    auws_.in2wo[input_index].emplace_back(weight_index, output_index);
  }
};

}  // namespace tiny_dnn
