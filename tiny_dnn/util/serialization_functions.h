/*
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <cereal/access.hpp>  // For LoadAndConstruct
#include "tiny_dnn/tiny_dnn.h"

namespace cereal {

template <>
struct LoadAndConstruct<tiny_dnn::elementwise_add_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar,
    cereal::construct<tiny_dnn::elementwise_add_layer> &construct) {
    tiny_dnn::serial_size_t num_args, dim;

    ar(cereal::make_nvp("num_args", num_args), cereal::make_nvp("dim", dim));
    construct(num_args, dim);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::average_pooling_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar,
    cereal::construct<tiny_dnn::average_pooling_layer> &construct) {
    tiny_dnn::shape3d in;
    tiny_dnn::serial_size_t stride_x, stride_y, pool_size_x, pool_size_y;
    tiny_dnn::padding pad_type;

    ar(cereal::make_nvp("in_size", in),
       cereal::make_nvp("pool_size_x", pool_size_x),
       cereal::make_nvp("pool_size_y", pool_size_y),
       cereal::make_nvp("stride_x", stride_x),
       cereal::make_nvp("stride_y", stride_y),
       cereal::make_nvp("pad_type", pad_type));
    construct(in.width_, in.height_, in.depth_, pool_size_x, pool_size_y,
              stride_x, stride_y, pad_type);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::average_unpooling_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar,
    cereal::construct<tiny_dnn::average_unpooling_layer> &construct) {
    tiny_dnn::shape3d in;
    tiny_dnn::serial_size_t pool_size, stride;

    ar(cereal::make_nvp("in_size", in),
       cereal::make_nvp("pool_size", pool_size),
       cereal::make_nvp("stride", stride));
    construct(in.width_, in.height_, in.depth_, pool_size, stride);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::batch_normalization_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar,
    cereal::construct<tiny_dnn::batch_normalization_layer> &construct) {
    tiny_dnn::shape3d in;
    tiny_dnn::serial_size_t in_spatial_size, in_channels;
    tiny_dnn::float_t eps, momentum;
    tiny_dnn::net_phase phase;
    tiny_dnn::vec_t mean, variance;

    ar(cereal::make_nvp("in_spatial_size", in_spatial_size),
       cereal::make_nvp("in_channels", in_channels),
       cereal::make_nvp("epsilon", eps), cereal::make_nvp("momentum", momentum),
       cereal::make_nvp("phase", phase), cereal::make_nvp("mean", mean),
       cereal::make_nvp("variance", variance));
    construct(in_spatial_size, in_channels, eps, momentum, phase);
    construct->set_mean(mean);
    construct->set_variance(variance);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::concat_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar, cereal::construct<tiny_dnn::concat_layer> &construct) {
    std::vector<tiny_dnn::shape3d> in_shapes;

    ar(cereal::make_nvp("in_size", in_shapes));
    construct(in_shapes);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::convolutional_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar, cereal::construct<tiny_dnn::convolutional_layer> &construct) {
    tiny_dnn::serial_size_t w_width, w_height, out_ch, w_stride, h_stride;
    bool has_bias;
    tiny_dnn::shape3d in;
    tiny_dnn::padding pad_type;
    tiny_dnn::core::connection_table tbl;

    ar(cereal::make_nvp("in_size", in),
       cereal::make_nvp("window_width", w_width),
       cereal::make_nvp("window_height", w_height),
       cereal::make_nvp("out_channels", out_ch),
       cereal::make_nvp("connection_table", tbl),
       cereal::make_nvp("pad_type", pad_type),
       cereal::make_nvp("has_bias", has_bias),
       cereal::make_nvp("w_stride", w_stride),
       cereal::make_nvp("h_stride", h_stride));

    construct(in.width_, in.height_, w_width, w_height, in.depth_, out_ch, tbl,
              pad_type, has_bias, w_stride, h_stride);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::deconvolutional_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar,
    cereal::construct<tiny_dnn::deconvolutional_layer> &construct) {
    tiny_dnn::serial_size_t w_width, w_height, out_ch, w_stride, h_stride;
    bool has_bias;
    tiny_dnn::shape3d in;
    tiny_dnn::padding pad_type;
    tiny_dnn::core::connection_table tbl;

    ar(cereal::make_nvp("in_size", in),
       cereal::make_nvp("window_width", w_width),
       cereal::make_nvp("window_height", w_height),
       cereal::make_nvp("out_channels", out_ch),
       cereal::make_nvp("connection_table", tbl),
       cereal::make_nvp("pad_type", pad_type),
       cereal::make_nvp("has_bias", has_bias),
       cereal::make_nvp("w_stride", w_stride),
       cereal::make_nvp("h_stride", h_stride));

    construct(in.width_, in.height_, w_width, w_height, in.depth_, out_ch, tbl,
              pad_type, has_bias, w_stride, h_stride);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::dropout_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar, cereal::construct<tiny_dnn::dropout_layer> &construct) {
    tiny_dnn::net_phase phase;
    tiny_dnn::float_t dropout_rate;
    tiny_dnn::serial_size_t in_size;

    ar(cereal::make_nvp("in_size", in_size),
       cereal::make_nvp("dropout_rate", dropout_rate),
       cereal::make_nvp("phase", phase));
    construct(in_size, dropout_rate, phase);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::fully_connected_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar,
    cereal::construct<tiny_dnn::fully_connected_layer> &construct) {
    tiny_dnn::serial_size_t in_dim, out_dim;
    bool has_bias;

    ar(cereal::make_nvp("in_size", in_dim),
       cereal::make_nvp("out_size", out_dim),
       cereal::make_nvp("has_bias", has_bias));
    construct(in_dim, out_dim, has_bias);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::global_average_pooling_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar,
    cereal::construct<tiny_dnn::global_average_pooling_layer> &construct) {
    tiny_dnn::shape3d in_shape;

    ar(cereal::make_nvp("in_shape", in_shape));
    construct(in_shape);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::input_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar, cereal::construct<tiny_dnn::input_layer> &construct) {
    tiny_dnn::shape3d shape;

    ar(cereal::make_nvp("shape", shape));
    construct(shape);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::linear_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar, cereal::construct<tiny_dnn::linear_layer> &construct) {
    tiny_dnn::serial_size_t dim;
    tiny_dnn::float_t scale, bias;

    ar(cereal::make_nvp("in_size", dim), cereal::make_nvp("scale", scale),
       cereal::make_nvp("bias", bias));

    construct(dim, scale, bias);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::lrn_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar, cereal::construct<tiny_dnn::lrn_layer> &construct) {
    tiny_dnn::shape3d in_shape;
    tiny_dnn::serial_size_t size;
    tiny_dnn::float_t alpha, beta;
    tiny_dnn::norm_region region;

    ar(cereal::make_nvp("in_shape", in_shape), cereal::make_nvp("size", size),
       cereal::make_nvp("alpha", alpha), cereal::make_nvp("beta", beta),
       cereal::make_nvp("region", region));
    construct(in_shape, size, alpha, beta, region);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::max_pooling_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar, cereal::construct<tiny_dnn::max_pooling_layer> &construct) {
    tiny_dnn::shape3d in;
    tiny_dnn::serial_size_t stride_x, stride_y, pool_size_x, pool_size_y;
    tiny_dnn::padding pad_type;

    ar(cereal::make_nvp("in_size", in),
       cereal::make_nvp("pool_size_x", pool_size_x),
       cereal::make_nvp("pool_size_y", pool_size_y),
       cereal::make_nvp("stride_x", stride_x),
       cereal::make_nvp("stride_y", stride_y),
       cereal::make_nvp("pad_type", pad_type));
    construct(in.width_, in.height_, in.depth_, pool_size_x, pool_size_y,
              stride_x, stride_y, pad_type);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::max_unpooling_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar, cereal::construct<tiny_dnn::max_unpooling_layer> &construct) {
    tiny_dnn::shape3d in;
    tiny_dnn::serial_size_t stride, unpool_size;

    ar(cereal::make_nvp("in_size", in),
       cereal::make_nvp("unpool_size", unpool_size),
       cereal::make_nvp("stride", stride));
    construct(in, unpool_size, stride);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::power_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar, cereal::construct<tiny_dnn::power_layer> &construct) {
    tiny_dnn::shape3d in_shape;
    tiny_dnn::float_t factor;
    tiny_dnn::float_t scale(1.0f);

    ar(cereal::make_nvp("in_size", in_shape),
       cereal::make_nvp("factor", factor), cereal::make_nvp("scale", scale));
    construct(in_shape, factor, scale);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::quantized_convolutional_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar,
    cereal::construct<tiny_dnn::quantized_convolutional_layer> &construct) {
    tiny_dnn::serial_size_t w_width, w_height, out_ch, w_stride, h_stride;
    bool has_bias;
    tiny_dnn::shape3d in;
    tiny_dnn::padding pad_type;
    tiny_dnn::core::connection_table tbl;

    ar(cereal::make_nvp("in_size", in),
       cereal::make_nvp("window_width", w_width),
       cereal::make_nvp("window_height", w_height),
       cereal::make_nvp("out_channels", out_ch),
       cereal::make_nvp("connection_table", tbl),
       cereal::make_nvp("pad_type", pad_type),
       cereal::make_nvp("has_bias", has_bias),
       cereal::make_nvp("w_stride", w_stride),
       cereal::make_nvp("h_stride", h_stride));

    construct(in.width_, in.height_, w_width, w_height, in.depth_, out_ch, tbl,
              pad_type, has_bias, w_stride, h_stride);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::quantized_deconvolutional_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar,
    cereal::construct<tiny_dnn::quantized_deconvolutional_layer> &construct) {
    tiny_dnn::serial_size_t w_width, w_height, out_ch, w_stride, h_stride;
    bool has_bias;
    tiny_dnn::shape3d in;
    tiny_dnn::padding pad_type;
    tiny_dnn::core::connection_table tbl;

    ar(cereal::make_nvp("in_size", in),
       cereal::make_nvp("window_width", w_width),
       cereal::make_nvp("window_height", w_height),
       cereal::make_nvp("out_channels", out_ch),
       cereal::make_nvp("connection_table", tbl),
       cereal::make_nvp("pad_type", pad_type),
       cereal::make_nvp("has_bias", has_bias),
       cereal::make_nvp("w_stride", w_stride),
       cereal::make_nvp("h_stride", h_stride));

    construct(in.width_, in.height_, w_width, w_height, in.depth_, out_ch, tbl,
              pad_type, has_bias, w_stride, h_stride);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::quantized_fully_connected_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar,
    cereal::construct<tiny_dnn::quantized_fully_connected_layer> &construct) {
    tiny_dnn::serial_size_t in_dim, out_dim;
    bool has_bias;

    ar(cereal::make_nvp("in_size", in_dim),
       cereal::make_nvp("out_size", out_dim),
       cereal::make_nvp("has_bias", has_bias));
    construct(in_dim, out_dim, has_bias);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::slice_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar, cereal::construct<tiny_dnn::slice_layer> &construct) {
    tiny_dnn::shape3d in_shape;
    tiny_dnn::slice_type slice_type;
    tiny_dnn::serial_size_t num_outputs;

    ar(cereal::make_nvp("in_size", in_shape),
       cereal::make_nvp("slice_type", slice_type),
       cereal::make_nvp("num_outputs", num_outputs));
    construct(in_shape, slice_type, num_outputs);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::sigmoid_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar, cereal::construct<tiny_dnn::sigmoid_layer> &construct) {
    tiny_dnn::shape3d in_shape;

    ar(cereal::make_nvp("in_size", in_shape));
    construct(in_shape);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::tanh_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar, cereal::construct<tiny_dnn::tanh_layer> &construct) {
    tiny_dnn::shape3d in_shape;

    ar(cereal::make_nvp("in_size", in_shape));
    construct(in_shape);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::relu_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar, cereal::construct<tiny_dnn::relu_layer> &construct) {
    tiny_dnn::shape3d in_shape;

    ar(cereal::make_nvp("in_size", in_shape));
    construct(in_shape);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::softmax_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar, cereal::construct<tiny_dnn::softmax_layer> &construct) {
    tiny_dnn::shape3d in_shape;

    ar(cereal::make_nvp("in_size", in_shape));
    construct(in_shape);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::leaky_relu_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar, cereal::construct<tiny_dnn::leaky_relu_layer> &construct) {
    tiny_dnn::shape3d in_shape;
    tiny_dnn::float_t epsilon;

    ar(cereal::make_nvp("in_size", in_shape),
       cereal::make_nvp("epsilon", epsilon));
    construct(in_shape, epsilon);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::elu_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar, cereal::construct<tiny_dnn::elu_layer> &construct) {
    tiny_dnn::shape3d in_shape;

    ar(cereal::make_nvp("in_size", in_shape));
    construct(in_shape);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::tanh_p1m2_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar, cereal::construct<tiny_dnn::tanh_p1m2_layer> &construct) {
    tiny_dnn::shape3d in_shape;

    ar(cereal::make_nvp("in_size", in_shape));
    construct(in_shape);
  }
};

template <class Archive>
struct specialize<Archive,
                  tiny_dnn::elementwise_add_layer,
                  cereal::specialization::non_member_serialize> {};

template <class Archive>
struct specialize<Archive,
                  tiny_dnn::average_pooling_layer,
                  cereal::specialization::non_member_serialize> {};

template <class Archive>
struct specialize<Archive,
                  tiny_dnn::average_unpooling_layer,
                  cereal::specialization::non_member_serialize> {};

template <class Archive>
struct specialize<Archive,
                  tiny_dnn::batch_normalization_layer,
                  cereal::specialization::non_member_serialize> {};

template <class Archive>
struct specialize<Archive,
                  tiny_dnn::concat_layer,
                  cereal::specialization::non_member_serialize> {};

template <class Archive>
struct specialize<Archive,
                  tiny_dnn::convolutional_layer,
                  cereal::specialization::non_member_serialize> {};

template <class Archive>
struct specialize<Archive,
                  tiny_dnn::deconvolutional_layer,
                  cereal::specialization::non_member_serialize> {};

template <class Archive>
struct specialize<Archive,
                  tiny_dnn::dropout_layer,
                  cereal::specialization::non_member_serialize> {};

template <class Archive>
struct specialize<Archive,
                  tiny_dnn::fully_connected_layer,
                  cereal::specialization::non_member_serialize> {};

template <class Archive>
struct specialize<Archive,
                  tiny_dnn::global_average_pooling_layer,
                  cereal::specialization::non_member_serialize> {};

template <class Archive>
struct specialize<Archive,
                  tiny_dnn::input_layer,
                  cereal::specialization::non_member_serialize> {};

template <class Archive>
struct specialize<Archive,
                  tiny_dnn::linear_layer,
                  cereal::specialization::non_member_serialize> {};

template <class Archive>
struct specialize<Archive,
                  tiny_dnn::lrn_layer,
                  cereal::specialization::non_member_serialize> {};

template <class Archive>
struct specialize<Archive,
                  tiny_dnn::max_pooling_layer,
                  cereal::specialization::non_member_serialize> {};

template <class Archive>
struct specialize<Archive,
                  tiny_dnn::max_unpooling_layer,
                  cereal::specialization::non_member_serialize> {};

template <class Archive>
struct specialize<Archive,
                  tiny_dnn::power_layer,
                  cereal::specialization::non_member_serialize> {};

template <class Archive>
struct specialize<Archive,
                  tiny_dnn::quantized_convolutional_layer,
                  cereal::specialization::non_member_serialize> {};

template <class Archive>
struct specialize<Archive,
                  tiny_dnn::quantized_deconvolutional_layer,
                  cereal::specialization::non_member_serialize> {};

template <class Archive>
struct specialize<Archive,
                  tiny_dnn::quantized_fully_connected_layer,
                  cereal::specialization::non_member_serialize> {};

template <class Archive>
struct specialize<Archive,
                  tiny_dnn::slice_layer,
                  cereal::specialization::non_member_serialize> {};

template <class Archive>
struct specialize<Archive,
                  tiny_dnn::sigmoid_layer,
                  cereal::specialization::non_member_serialize> {};

template <class Archive>
struct specialize<Archive,
                  tiny_dnn::tanh_layer,
                  cereal::specialization::non_member_serialize> {};

template <class Archive>
struct specialize<Archive,
                  tiny_dnn::relu_layer,
                  cereal::specialization::non_member_serialize> {};

template <class Archive>
struct specialize<Archive,
                  tiny_dnn::softmax_layer,
                  cereal::specialization::non_member_serialize> {};

template <class Archive>
struct specialize<Archive,
                  tiny_dnn::leaky_relu_layer,
                  cereal::specialization::non_member_serialize> {};

template <class Archive>
struct specialize<Archive,
                  tiny_dnn::elu_layer,
                  cereal::specialization::non_member_serialize> {};

template <class Archive>
struct specialize<Archive,
                  tiny_dnn::tanh_p1m2_layer,
                  cereal::specialization::non_member_serialize> {};

}  // namespace cereal

namespace tiny_dnn {

struct serialization_buddy {
#ifndef CNN_NO_SERIALIZATION
  template <class Archive>
  static inline void serialize(Archive &ar,
                               tiny_dnn::elementwise_add_layer &layer) {
    layer.serialize_prolog(ar);
    ar(cereal::make_nvp("num_args", layer.num_args_),
       cereal::make_nvp("dim", layer.dim_));
  }

  template <class Archive>
  static inline void serialize(Archive &ar,
                               tiny_dnn::average_pooling_layer &layer) {
    layer.serialize_prolog(ar);
    ar(cereal::make_nvp("in_size", layer.in_),
       cereal::make_nvp("pool_size_x", layer.pool_size_x_),
       cereal::make_nvp("pool_size_y", layer.pool_size_y_),
       cereal::make_nvp("stride_x", layer.stride_x_),
       cereal::make_nvp("stride_y", layer.stride_y_),
       cereal::make_nvp("pad_type", layer.pad_type_));
  }

  template <class Archive>
  static inline void serialize(Archive &ar,
                               tiny_dnn::average_unpooling_layer &layer) {
    layer.serialize_prolog(ar);
    ar(cereal::make_nvp("in_size", layer.in_),
       cereal::make_nvp("pool_size", layer.w_.width_),
       cereal::make_nvp("stride", layer.stride_));
  }

  template <class Archive>
  static inline void serialize(Archive &ar,
                               tiny_dnn::batch_normalization_layer &layer) {
    layer.serialize_prolog(ar);
    ar(cereal::make_nvp("in_spatial_size", layer.in_spatial_size_),
       cereal::make_nvp("in_channels", layer.in_channels_),
       cereal::make_nvp("epsilon", layer.eps_),
       cereal::make_nvp("momentum", layer.momentum_),
       cereal::make_nvp("phase", layer.phase_),
       cereal::make_nvp("mean", layer.mean_),
       cereal::make_nvp("variance", layer.variance_));
  }

  template <class Archive>
  static inline void serialize(Archive &ar, tiny_dnn::concat_layer &layer) {
    layer.serialize_prolog(ar);
    ar(layer.in_shapes_);
  }

  template <class Archive>
  static inline void serialize(Archive &ar,
                               tiny_dnn::convolutional_layer &layer) {
    layer.serialize_prolog(ar);
    auto &params_ = layer.params_;
    ar(cereal::make_nvp("in_size", params_.in),
       cereal::make_nvp("window_width", params_.weight.width_),
       cereal::make_nvp("window_height", params_.weight.height_),
       cereal::make_nvp("out_channels", params_.out.depth_),
       cereal::make_nvp("connection_table", params_.tbl),
       cereal::make_nvp("pad_type", params_.pad_type),
       cereal::make_nvp("has_bias", params_.has_bias),
       cereal::make_nvp("w_stride", params_.w_stride),
       cereal::make_nvp("h_stride", params_.h_stride));
  }

  template <class Archive>
  static inline void serialize(Archive &ar,
                               tiny_dnn::deconvolutional_layer &layer) {
    layer.serialize_prolog(ar);
    auto &params_ = layer.params_;
    ar(cereal::make_nvp("in_size", params_.in),
       cereal::make_nvp("window_width", params_.weight.width_),
       cereal::make_nvp("window_height", params_.weight.height_),
       cereal::make_nvp("out_channels", params_.out.depth_),
       cereal::make_nvp("connection_table", params_.tbl),
       cereal::make_nvp("pad_type", params_.pad_type),
       cereal::make_nvp("has_bias", params_.has_bias),
       cereal::make_nvp("w_stride", params_.w_stride),
       cereal::make_nvp("h_stride", params_.h_stride));
  }

  template <class Archive>
  static inline void serialize(Archive &ar, tiny_dnn::dropout_layer &layer) {
    layer.serialize_prolog(ar);
    ar(cereal::make_nvp("in_size", layer.in_size_),
       cereal::make_nvp("dropout_rate", layer.dropout_rate_),
       cereal::make_nvp("phase", layer.phase_));
  }

  template <class Archive>
  static inline void serialize(Archive &ar,
                               tiny_dnn::fully_connected_layer &layer) {
    layer.serialize_prolog(ar);
    auto &params_ = layer.params_;
    ar(cereal::make_nvp("in_size", params_.in_size_),
       cereal::make_nvp("out_size", params_.out_size_),
       cereal::make_nvp("has_bias", params_.has_bias_));
  }

  template <class Archive>
  static inline void serialize(Archive &ar,
                               tiny_dnn::global_average_pooling_layer &layer) {
    layer.serialize_prolog(ar);
    auto &params_ = layer.params_;
    ar(cereal::make_nvp("in_shape", params_.in));
  }

  template <class Archive>
  static inline void serialize(Archive &ar, tiny_dnn::input_layer &layer) {
    layer.serialize_prolog(ar);
    ar(cereal::make_nvp("shape", layer.shape_));
  }

  template <class Archive>
  static inline void serialize(Archive &ar, tiny_dnn::linear_layer &layer) {
    layer.serialize_prolog(ar);
    ar(cereal::make_nvp("in_size", layer.dim_),
       cereal::make_nvp("scale", layer.scale_),
       cereal::make_nvp("bias", layer.bias_));
  }

  template <class Archive>
  static inline void serialize(Archive &ar, tiny_dnn::lrn_layer &layer) {
    layer.serialize_prolog(ar);
    ar(cereal::make_nvp("in_shape", layer.in_shape_),
       cereal::make_nvp("size", layer.size_),
       cereal::make_nvp("alpha", layer.alpha_),
       cereal::make_nvp("beta", layer.beta_),
       cereal::make_nvp("region", layer.region_));
  }

  template <class Archive>
  static inline void serialize(Archive &ar,
                               tiny_dnn::max_pooling_layer &layer) {
    layer.serialize_prolog(ar);
    auto &params_ = layer.params_;
    ar(cereal::make_nvp("in_size", params_.in),
       cereal::make_nvp("pool_size_x", params_.pool_size_x),
       cereal::make_nvp("pool_size_y", params_.pool_size_y),
       cereal::make_nvp("stride_x", params_.stride_x),
       cereal::make_nvp("stride_y", params_.stride_y),
       cereal::make_nvp("pad_type", params_.pad_type));
  }

  template <class Archive>
  static inline void serialize(Archive &ar,
                               tiny_dnn::max_unpooling_layer &layer) {
    layer.serialize_prolog(ar);
    ar(cereal::make_nvp("in_size", layer.in_),
       cereal::make_nvp("unpool_size", layer.unpool_size_),
       cereal::make_nvp("stride", layer.stride_));
  }

  template <class Archive>
  static inline void serialize(Archive &ar, tiny_dnn::power_layer &layer) {
    layer.serialize_prolog(ar);
    ar(cereal::make_nvp("in_size", layer.in_shape_),
       cereal::make_nvp("factor", layer.factor_),
       cereal::make_nvp("scale", layer.scale_));
  }

  template <class Archive>
  static inline void serialize(Archive &ar,
                               tiny_dnn::quantized_convolutional_layer &layer) {
    layer.serialize_prolog(ar);
    auto &params_ = layer.params_;
    ar(cereal::make_nvp("in_size", params_.in),
       cereal::make_nvp("window_width", params_.weight.width_),
       cereal::make_nvp("window_height", params_.weight.height_),
       cereal::make_nvp("out_channels", params_.out.depth_),
       cereal::make_nvp("connection_table", params_.tbl),
       cereal::make_nvp("pad_type", params_.pad_type),
       cereal::make_nvp("has_bias", params_.has_bias),
       cereal::make_nvp("w_stride", params_.w_stride),
       cereal::make_nvp("h_stride", params_.h_stride));
  }

  template <class Archive>
  static inline void serialize(
    Archive &ar, tiny_dnn::quantized_deconvolutional_layer &layer) {
    layer.serialize_prolog(ar);
    auto &params_ = layer.params_;
    ar(cereal::make_nvp("in_size", params_.in),
       cereal::make_nvp("window_width", params_.weight.width_),
       cereal::make_nvp("window_height", params_.weight.height_),
       cereal::make_nvp("out_channels", params_.out.depth_),
       cereal::make_nvp("connection_table", params_.tbl),
       cereal::make_nvp("pad_type", params_.pad_type),
       cereal::make_nvp("has_bias", params_.has_bias),
       cereal::make_nvp("w_stride", params_.w_stride),
       cereal::make_nvp("h_stride", params_.h_stride));
  }

  template <class Archive>
  static inline void serialize(
    Archive &ar, tiny_dnn::quantized_fully_connected_layer &layer) {
    layer.serialize_prolog(ar);
    auto &params_ = layer.params_;
    ar(cereal::make_nvp("in_size", params_.in_size_),
       cereal::make_nvp("out_size", params_.out_size_),
       cereal::make_nvp("has_bias", params_.has_bias_));
  }

  template <class Archive>
  static inline void serialize(Archive &ar, tiny_dnn::slice_layer &layer) {
    layer.serialize_prolog(ar);
    ar(cereal::make_nvp("in_size", layer.in_shape_),
       cereal::make_nvp("slice_type", layer.slice_type_),
       cereal::make_nvp("num_outputs", layer.num_outputs_));
  }

  template <class Archive>
  static inline void serialize(Archive &ar, tiny_dnn::sigmoid_layer &layer) {
    layer.serialize_prolog(ar);
    ar(cereal::make_nvp("in_size", layer.in_shape()[0]));
  }

  template <class Archive>
  static inline void serialize(Archive &ar, tiny_dnn::tanh_layer &layer) {
    layer.serialize_prolog(ar);
    ar(cereal::make_nvp("in_size", layer.in_shape()[0]));
  }

  template <class Archive>
  static inline void serialize(Archive &ar, tiny_dnn::relu_layer &layer) {
    layer.serialize_prolog(ar);
    ar(cereal::make_nvp("in_size", layer.in_shape()[0]));
  }

  template <class Archive>
  static inline void serialize(Archive &ar, tiny_dnn::softmax_layer &layer) {
    layer.serialize_prolog(ar);
    ar(cereal::make_nvp("in_size", layer.in_shape()[0]));
  }

  template <class Archive>
  static inline void serialize(Archive &ar, tiny_dnn::leaky_relu_layer &layer) {
    layer.serialize_prolog(ar);
    ar(cereal::make_nvp("in_size", layer.in_shape()[0]),
       cereal::make_nvp("epsilon", layer.epsilon_));
  }

  template <class Archive>
  static inline void serialize(Archive &ar, tiny_dnn::elu_layer &layer) {
    layer.serialize_prolog(ar);
    ar(cereal::make_nvp("in_size", layer.in_shape()[0]));
  }

  template <class Archive>
  static inline void serialize(Archive &ar, tiny_dnn::tanh_p1m2_layer &layer) {
    layer.serialize_prolog(ar);
    ar(cereal::make_nvp("in_size", layer.in_shape()[0]));
  }
#endif
};

template <class Archive>
void serialize(Archive &ar, tiny_dnn::elementwise_add_layer &layer) {
  serialization_buddy::serialize(ar, layer);
}

template <class Archive>
void serialize(Archive &ar, tiny_dnn::average_pooling_layer &layer) {
  serialization_buddy::serialize(ar, layer);
}

template <class Archive>
void serialize(Archive &ar, tiny_dnn::average_unpooling_layer &layer) {
  serialization_buddy::serialize(ar, layer);
}

template <class Archive>
void serialize(Archive &ar, tiny_dnn::batch_normalization_layer &layer) {
  serialization_buddy::serialize(ar, layer);
}

template <class Archive>
void serialize(Archive &ar, tiny_dnn::concat_layer &layer) {
  serialization_buddy::serialize(ar, layer);
}

template <class Archive>
void serialize(Archive &ar, tiny_dnn::convolutional_layer &layer) {
  serialization_buddy::serialize(ar, layer);
}

template <class Archive>
void serialize(Archive &ar, tiny_dnn::deconvolutional_layer &layer) {
  serialization_buddy::serialize(ar, layer);
}

template <class Archive>
void serialize(Archive &ar, tiny_dnn::dropout_layer &layer) {
  serialization_buddy::serialize(ar, layer);
}

template <class Archive>
void serialize(Archive &ar, tiny_dnn::fully_connected_layer &layer) {
  serialization_buddy::serialize(ar, layer);
}

template <class Archive>
void serialize(Archive &ar, tiny_dnn::global_average_pooling_layer &layer) {
  serialization_buddy::serialize(ar, layer);
}

template <class Archive>
void serialize(Archive &ar, tiny_dnn::input_layer &layer) {
  serialization_buddy::serialize(ar, layer);
}

template <class Archive>
void serialize(Archive &ar, tiny_dnn::linear_layer &layer) {
  serialization_buddy::serialize(ar, layer);
}

template <class Archive>
void serialize(Archive &ar, tiny_dnn::lrn_layer &layer) {
  serialization_buddy::serialize(ar, layer);
}

template <class Archive>
void serialize(Archive &ar, tiny_dnn::max_pooling_layer &layer) {
  serialization_buddy::serialize(ar, layer);
}

template <class Archive>
void serialize(Archive &ar, tiny_dnn::max_unpooling_layer &layer) {
  serialization_buddy::serialize(ar, layer);
}

template <class Archive>
void serialize(Archive &ar, tiny_dnn::power_layer &layer) {
  serialization_buddy::serialize(ar, layer);
}

template <class Archive>
void serialize(Archive &ar, tiny_dnn::quantized_convolutional_layer &layer) {
  serialization_buddy::serialize(ar, layer);
}

template <class Archive>
void serialize(Archive &ar, tiny_dnn::quantized_deconvolutional_layer &layer) {
  serialization_buddy::serialize(ar, layer);
}

template <class Archive>
void serialize(Archive &ar, tiny_dnn::quantized_fully_connected_layer &layer) {
  serialization_buddy::serialize(ar, layer);
}

template <class Archive>
void serialize(Archive &ar, tiny_dnn::slice_layer &layer) {
  serialization_buddy::serialize(ar, layer);
}

template <class Archive>
void serialize(Archive &ar, tiny_dnn::sigmoid_layer &layer) {
  serialization_buddy::serialize(ar, layer);
}

template <class Archive>
void serialize(Archive &ar, tiny_dnn::tanh_layer &layer) {
  serialization_buddy::serialize(ar, layer);
}

template <class Archive>
void serialize(Archive &ar, tiny_dnn::relu_layer &layer) {
  serialization_buddy::serialize(ar, layer);
}

template <class Archive>
void serialize(Archive &ar, tiny_dnn::softmax_layer &layer) {
  serialization_buddy::serialize(ar, layer);
}

template <class Archive>
void serialize(Archive &ar, tiny_dnn::leaky_relu_layer &layer) {
  serialization_buddy::serialize(ar, layer);
}

template <class Archive>
void serialize(Archive &ar, tiny_dnn::elu_layer &layer) {
  serialization_buddy::serialize(ar, layer);
}

template <class Archive>
void serialize(Archive &ar, tiny_dnn::tanh_p1m2_layer &layer) {
  serialization_buddy::serialize(ar, layer);
}

template <class Archive, typename T>
void serialize(Archive &ar, tiny_dnn::index3d<T> &idx) {
  ar(cereal::make_nvp("width", idx.width_),
     cereal::make_nvp("height", idx.height_),
     cereal::make_nvp("depth", idx.depth_));
}

namespace core {

template <class Archive>
void serialize(Archive &ar, tiny_dnn::core::connection_table &tbl) {
  ar(cereal::make_nvp("rows", tbl.rows_), cereal::make_nvp("cols", tbl.cols_));
  if (tbl.is_empty()) {
    ar(cereal::make_nvp("connection", std::string("all")));
  } else {
    ar(cereal::make_nvp("connection", tbl.connected_));
  }
}

}  // namespace core

}  // namespace tiny_dnn
