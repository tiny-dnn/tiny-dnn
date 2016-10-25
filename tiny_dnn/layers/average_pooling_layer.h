// Copyright (c) 2013-2016, Taiga Nomi. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "tiny_dnn/activations/activation_function.h"
#include "tiny_dnn/layers/partial_connected_layer.h"
#include "tiny_dnn/util/image.h"
#include "tiny_dnn/util/util.h"

namespace tiny_dnn {

// forward_propagation
template <typename Activation>
void tiny_average_pooling_kernel(
    bool parallelize, const std::vector<tensor_t*>& in_data,
    std::vector<tensor_t*>& out_data, const shape3d& out_dim,
    float_t scale_factor,
    std::vector<typename partial_connected_layer<Activation>::wi_connections>&
        out2wi,
    Activation& h) {
  for_i(in_data[0]->size(), [&](size_t sample) {
    const vec_t& in = (*in_data[0])[sample];
    const vec_t& W = (*in_data[1])[0];
    const vec_t& b = (*in_data[2])[0];
    vec_t& out = (*out_data[0])[sample];
    vec_t& a = (*out_data[1])[sample];

    auto oarea = out_dim.area();
    size_t idx = 0;
    for (size_t d = 0; d < out_dim.depth_; ++d) {
      float_t weight = W[d] * scale_factor;
      float_t bias = b[d];
      for (size_t i = 0; i < oarea; ++i, ++idx) {
        const auto& connections = out2wi[idx];
        float_t value = float_t(0);
        for (auto connection : connections)  // 13.1%
          value += in[connection.second];    // 3.2%
        value *= weight;
        value += bias;
        a[idx] = value;
      }
    }

    assert(out.size() == out2wi.size());
    for (size_t i = 0; i < out2wi.size(); i++) {
      out[i] = h.f(a, i);
    }
  });
}

// back_propagation
template <typename Activation>
void tiny_average_pooling_back_kernel(
    const std::vector<tensor_t*>& in_data,
    const std::vector<tensor_t*>& out_data, std::vector<tensor_t*>& out_grad,
    std::vector<tensor_t*>& in_grad, const shape3d& in_dim,
    float_t scale_factor,
    std::vector<typename partial_connected_layer<Activation>::io_connections>&
        weight2io,
    std::vector<typename partial_connected_layer<Activation>::wo_connections>&
        in2wo,
    std::vector<std::vector<cnn_size_t>>& bias2out) {
  for_i(in_data[0]->size(), [&](size_t sample) {
    const vec_t& prev_out = (*in_data[0])[sample];
    const vec_t& W = (*in_data[1])[0];
    vec_t& dW = (*in_grad[1])[sample];
    vec_t& db = (*in_grad[2])[sample];
    vec_t& prev_delta = (*in_grad[0])[sample];
    vec_t& curr_delta = (*out_grad[0])[sample];

    auto inarea = in_dim.area();
    size_t idx = 0;
    for (size_t i = 0; i < in_dim.depth_; ++i) {
      float_t weight = W[i] * scale_factor;
      for (size_t j = 0; j < inarea; ++j, ++idx) {
        prev_delta[idx] = weight * curr_delta[in2wo[idx][0].second];
      }
    }

    for (size_t i = 0; i < weight2io.size(); ++i) {
      const auto& connections = weight2io[i];
      float_t diff = float_t(0);

      for (auto connection : connections)
        diff += prev_out[connection.first] * curr_delta[connection.second];

      dW[i] += diff * scale_factor;
    }

    for (size_t i = 0; i < bias2out.size(); i++) {
      const std::vector<cnn_size_t>& outs = bias2out[i];
      float_t diff = float_t(0);

      for (auto o : outs) diff += curr_delta[o];

      db[i] += diff;
    }
  });
}

/**
 * average pooling with trainable weights
 **/
template <typename Activation = activation::identity>
class average_pooling_layer : public partial_connected_layer<Activation> {
 public:
  typedef partial_connected_layer<Activation> Base;
  CNN_USE_LAYER_MEMBERS;

  /**
   * @param in_width     [in] width of input image
   * @param in_height    [in] height of input image
   * @param in_channels  [in] the number of input image channels(depth)
   * @param pooling_size [in] factor by which to downscale
   **/
  average_pooling_layer(cnn_size_t in_width, cnn_size_t in_height,
                        cnn_size_t in_channels, cnn_size_t pooling_size)
      : average_pooling_layer(in_width, in_height, in_channels, pooling_size,
                              pooling_size) {}

  average_pooling_layer(const shape3d& in_shape, cnn_size_t pooling_size,
                        cnn_size_t stride)
      : average_pooling_layer(in_shape.width_, in_shape.width_, in_shape.depth_,
                              pooling_size, stride) {}

  /**
   * @param in_width     [in] width of input image
   * @param in_height    [in] height of input image
   * @param in_channels  [in] the number of input image channels(depth)
   * @param pool_size    [in] factor by which to downscale
   * @param stride       [in] interval at which to apply the filters to the
   *input
  **/
  average_pooling_layer(cnn_size_t in_width, cnn_size_t in_height,
                        cnn_size_t in_channels, cnn_size_t pool_size,
                        cnn_size_t stride)
      : Base(in_width * in_height * in_channels,
             pool_out_dim(in_width, pool_size, stride) *
                 pool_out_dim(in_height, pool_size, stride) * in_channels,
             in_channels, in_channels, float_t(1) / sqr(pool_size)),
        stride_(stride),
        pool_size_(pool_size),
        in_(in_width, in_height, in_channels),
        out_(pool_out_dim(in_width, pool_size, stride),
             pool_out_dim(in_height, pool_size, stride), in_channels),
        w_(pool_size, pool_size, in_channels) {
    if ((in_width % pool_size) || (in_height % pool_size)) {
      pooling_size_mismatch(in_width, in_height, pool_size);
    }

    init_connection(pool_size);
  }

  std::vector<index3d<cnn_size_t>> in_shape() const override {
    return {in_, w_, index3d<cnn_size_t>(1, 1, out_.depth_)};
  }

  std::vector<index3d<cnn_size_t>> out_shape() const override {
    return {out_, out_};
  }

  std::string layer_type() const override { return "ave-pool"; }

  void forward_propagation(const std::vector<tensor_t*>& in_data,
                           std::vector<tensor_t*>& out_data) override {
    tiny_average_pooling_kernel<Activation>(parallelize_, in_data, out_data,
                                            out_, Base::scale_factor_,
                                            Base::out2wi_, Base::h_);
  }

  void back_propagation(const std::vector<tensor_t*>& in_data,
                        const std::vector<tensor_t*>& out_data,
                        std::vector<tensor_t*>& out_grad,
                        std::vector<tensor_t*>& in_grad) override {
    tensor_t& curr_delta = *out_grad[0];
    this->backward_activation(*out_grad[0], *out_data[0], curr_delta);

    tiny_average_pooling_back_kernel<Activation>(
        in_data, out_data, out_grad, in_grad, in_, Base::scale_factor_,
        Base::weight2io_, Base::in2wo_, Base::bias2out_);
  }

  template <class Archive>
  static void load_and_construct(
      Archive& ar, cereal::construct<average_pooling_layer>& construct) {
    shape3d in;
    size_t stride, pool_size;

    ar(cereal::make_nvp("in_size", in),
       cereal::make_nvp("pool_size", pool_size),
       cereal::make_nvp("stride", stride));
    construct(in, pool_size, stride);
  }

  template <class Archive>
  void serialize(Archive& ar) {
    layer::serialize_prolog(ar);
    ar(cereal::make_nvp("in_size", in_),
       cereal::make_nvp("pool_size", pool_size_),
       cereal::make_nvp("stride", stride_));
  }

 private:
  size_t stride_;
  size_t pool_size_;
  shape3d in_;
  shape3d out_;
  shape3d w_;

  static cnn_size_t pool_out_dim(cnn_size_t in_size, cnn_size_t pooling_size,
                                 cnn_size_t stride) {
    return static_cast<int>(
        std::ceil((static_cast<float_t>(in_size) - pooling_size) / stride) + 1);
  }

  void init_connection(cnn_size_t pooling_size) {
    for (cnn_size_t c = 0; c < in_.depth_; ++c) {
      for (cnn_size_t y = 0; y < in_.height_ - pooling_size + 1; y += stride_) {
        for (cnn_size_t x = 0; x < in_.width_ - pooling_size + 1;
             x += stride_) {
          connect_kernel(pooling_size, x, y, c);
        }
      }
    }

    for (cnn_size_t c = 0; c < in_.depth_; ++c) {
      for (cnn_size_t y = 0; y < out_.height_; ++y) {
        for (cnn_size_t x = 0; x < out_.width_; ++x) {
          this->connect_bias(c, out_.get_index(x, y, c));
        }
      }
    }
  }

  void connect_kernel(cnn_size_t pooling_size, cnn_size_t x, cnn_size_t y,
                      cnn_size_t inc) {
    cnn_size_t dymax = std::min(pooling_size, in_.height_ - y);
    cnn_size_t dxmax = std::min(pooling_size, in_.width_ - x);
    cnn_size_t dstx = x / stride_;
    cnn_size_t dsty = y / stride_;
    cnn_size_t outidx = out_.get_index(dstx, dsty, inc);
    for (cnn_size_t dy = 0; dy < dymax; ++dy) {
      for (cnn_size_t dx = 0; dx < dxmax; ++dx) {
        this->connect_weight(in_.get_index(x + dx, y + dy, inc), outidx, inc);
      }
    }
  }
};

}  // namespace tiny_dnn
