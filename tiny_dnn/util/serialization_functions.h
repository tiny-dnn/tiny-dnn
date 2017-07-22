/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <string>
#include <utility>
#include <vector>

#include <cereal/access.hpp>  // For LoadAndConstruct
#include "tiny_dnn/tiny_dnn.h"

namespace detail {

/**
 * size of layer, model, data etc.
 * change to smaller type if memory footprint is severe
 **/
typedef std::uint32_t serial_size_t;

typedef tiny_dnn::index3d<serial_size_t> shape3d_serial;

template <class T>
static inline cereal::NameValuePair<T> make_nvp(const char *name, T &&value) {
  return cereal::make_nvp(name, value);
}

template <class Archive, typename T>
void arc(Archive &ar, cereal::NameValuePair<T> &&arg) {
  ar(arg);
}

template <class Archive, typename T>
void arc(Archive &ar, T &&arg) {
  ar(arg);
}

template <class Archive,
          typename std::enable_if<std::is_base_of<cereal::BinaryOutputArchive,
                                                  Archive>::value>::type = 0>
void arc(Archive &ar, cereal::NameValuePair<size_t> &&arg) {
  cereal::NameValuePair<serial_size_t> arg2(arg.name, arg.value);
  ar(arg2);
}

template <class Archive,
          typename std::enable_if<std::is_base_of<cereal::BinaryInputArchive,
                                                  Archive>::value>::type = 0>
void arc(Archive &ar, cereal::NameValuePair<size_t> &&arg) {
  cereal::NameValuePair<serial_size_t> arg2(arg.name, 0);
  ar(arg2);
  arg.value = arg2.value;
}

template <class Archive>
inline void arc(Archive &ar) {}

template <class Archive, class Type, class Type2>
inline void arc(Archive &ar, Type &&arg, Type2 &&arg2) {
  arc(ar, std::forward<Type>(arg));
  arc(ar, std::forward<Type2>(arg2));
}

template <class Archive, class Type, class... Types>
inline void arc(Archive &ar, Type &&arg, Types &&... args) {
  arc(ar, std::forward<Type>(arg));
  arc(ar, std::forward<Types>(args)...);
}

}  // namespace detail

namespace cereal {

template <>
struct LoadAndConstruct<tiny_dnn::elementwise_add_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar,
    cereal::construct<tiny_dnn::elementwise_add_layer> &construct) {
    size_t num_args, dim;

    ::detail::arc(ar, ::detail::make_nvp("num_args", num_args),
                  ::detail::make_nvp("dim", dim));
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
    size_t stride_x, stride_y, pool_size_x, pool_size_y;
    tiny_dnn::padding pad_type;

    ::detail::arc(ar, ::detail::make_nvp("in_size", in),
                  ::detail::make_nvp("pool_size_x", pool_size_x),
                  ::detail::make_nvp("pool_size_y", pool_size_y),
                  ::detail::make_nvp("stride_x", stride_x),
                  ::detail::make_nvp("stride_y", stride_y),
                  ::detail::make_nvp("pad_type", pad_type));
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
    size_t pool_size, stride;

    ::detail::arc(ar, ::detail::make_nvp("in_size", in),
                  ::detail::make_nvp("pool_size", pool_size),
                  ::detail::make_nvp("stride", stride));
    construct(in.width_, in.height_, in.depth_, pool_size, stride);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::batch_normalization_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar,
    cereal::construct<tiny_dnn::batch_normalization_layer> &construct) {
    size_t in_spatial_size, in_channels;
    tiny_dnn::float_t eps, momentum;
    tiny_dnn::net_phase phase;
    tiny_dnn::vec_t mean, variance;

    ::detail::arc(ar, ::detail::make_nvp("in_spatial_size", in_spatial_size),
                  ::detail::make_nvp("in_channels", in_channels),
                  ::detail::make_nvp("epsilon", eps),
                  ::detail::make_nvp("momentum", momentum),
                  ::detail::make_nvp("phase", phase),
                  ::detail::make_nvp("mean", mean),
                  ::detail::make_nvp("variance", variance));
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
    ::detail::arc(ar, ::detail::make_nvp("in_size", in_shapes));
    construct(in_shapes);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::convolutional_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar, cereal::construct<tiny_dnn::convolutional_layer> &construct) {
    size_t w_width, w_height, out_ch, w_stride, h_stride;
    bool has_bias;
    tiny_dnn::shape3d in;
    tiny_dnn::padding pad_type;
    tiny_dnn::core::connection_table tbl;

    ::detail::arc(ar, ::detail::make_nvp("in_size", in),
                  ::detail::make_nvp("window_width", w_width),
                  ::detail::make_nvp("window_height", w_height),
                  ::detail::make_nvp("out_channels", out_ch),
                  ::detail::make_nvp("connection_table", tbl),
                  ::detail::make_nvp("pad_type", pad_type),
                  ::detail::make_nvp("has_bias", has_bias),
                  ::detail::make_nvp("w_stride", w_stride),
                  ::detail::make_nvp("h_stride", h_stride));

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
    size_t w_width, w_height, out_ch, w_stride, h_stride;
    bool has_bias;
    tiny_dnn::shape3d in;
    tiny_dnn::padding pad_type;
    tiny_dnn::core::connection_table tbl;

    ::detail::arc(ar, ::detail::make_nvp("in_size", in),
                  ::detail::make_nvp("window_width", w_width),
                  ::detail::make_nvp("window_height", w_height),
                  ::detail::make_nvp("out_channels", out_ch),
                  ::detail::make_nvp("connection_table", tbl),
                  ::detail::make_nvp("pad_type", pad_type),
                  ::detail::make_nvp("has_bias", has_bias),
                  ::detail::make_nvp("w_stride", w_stride),
                  ::detail::make_nvp("h_stride", h_stride));

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
    size_t in_size;

    ::detail::arc(ar, ::detail::make_nvp("in_size", in_size),
                  ::detail::make_nvp("dropout_rate", dropout_rate),
                  ::detail::make_nvp("phase", phase));
    construct(in_size, dropout_rate, phase);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::fully_connected_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar,
    cereal::construct<tiny_dnn::fully_connected_layer> &construct) {
    size_t in_dim, out_dim;
    bool has_bias;

    ::detail::arc(ar, ::detail::make_nvp("in_size", in_dim),
                  ::detail::make_nvp("out_size", out_dim),
                  ::detail::make_nvp("has_bias", has_bias));
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

    ::detail::arc(ar, ::detail::make_nvp("in_shape", in_shape));
    construct(in_shape);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::input_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar, cereal::construct<tiny_dnn::input_layer> &construct) {
    tiny_dnn::shape3d shape;

    ::detail::arc(ar, ::detail::make_nvp("shape", shape));
    construct(shape);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::linear_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar, cereal::construct<tiny_dnn::linear_layer> &construct) {
    size_t dim;
    tiny_dnn::float_t scale, bias;

    ::detail::arc(ar, ::detail::make_nvp("in_size", dim),
                  ::detail::make_nvp("scale", scale),
                  ::detail::make_nvp("bias", bias));

    construct(dim, scale, bias);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::lrn_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar, cereal::construct<tiny_dnn::lrn_layer> &construct) {
    tiny_dnn::shape3d in_shape;
    size_t size;
    tiny_dnn::float_t alpha, beta;
    tiny_dnn::norm_region region;

    ::detail::arc(
      ar, ::detail::make_nvp("in_shape", in_shape),
      ::detail::make_nvp("size", size), ::detail::make_nvp("alpha", alpha),
      ::detail::make_nvp("beta", beta), ::detail::make_nvp("region", region));
    construct(in_shape, size, alpha, beta, region);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::max_pooling_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar, cereal::construct<tiny_dnn::max_pooling_layer> &construct) {
    tiny_dnn::shape3d in;
    size_t stride_x, stride_y, pool_size_x, pool_size_y;
    tiny_dnn::padding pad_type;

    ::detail::arc(ar, ::detail::make_nvp("in_size", in),
                  ::detail::make_nvp("pool_size_x", pool_size_x),
                  ::detail::make_nvp("pool_size_y", pool_size_y),
                  ::detail::make_nvp("stride_x", stride_x),
                  ::detail::make_nvp("stride_y", stride_y),
                  ::detail::make_nvp("pad_type", pad_type));
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
    size_t stride, unpool_size;

    ::detail::arc(ar, ::detail::make_nvp("in_size", in),
                  ::detail::make_nvp("unpool_size", unpool_size),
                  ::detail::make_nvp("stride", stride));
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

    ::detail::arc(ar, ::detail::make_nvp("in_size", in_shape),
                  ::detail::make_nvp("factor", factor),
                  ::detail::make_nvp("scale", scale));
    construct(in_shape, factor, scale);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::quantized_convolutional_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar,
    cereal::construct<tiny_dnn::quantized_convolutional_layer> &construct) {
    size_t w_width, w_height, out_ch, w_stride, h_stride;
    bool has_bias;
    tiny_dnn::shape3d in;
    tiny_dnn::padding pad_type;
    tiny_dnn::core::connection_table tbl;

    ::detail::arc(ar, ::detail::make_nvp("in_size", in),
                  ::detail::make_nvp("window_width", w_width),
                  ::detail::make_nvp("window_height", w_height),
                  ::detail::make_nvp("out_channels", out_ch),
                  ::detail::make_nvp("connection_table", tbl),
                  ::detail::make_nvp("pad_type", pad_type),
                  ::detail::make_nvp("has_bias", has_bias),
                  ::detail::make_nvp("w_stride", w_stride),
                  ::detail::make_nvp("h_stride", h_stride));

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
    size_t w_width, w_height, out_ch, w_stride, h_stride;
    bool has_bias;
    tiny_dnn::shape3d in;
    tiny_dnn::padding pad_type;
    tiny_dnn::core::connection_table tbl;

    ::detail::arc(ar, ::detail::make_nvp("in_size", in),
                  ::detail::make_nvp("window_width", w_width),
                  ::detail::make_nvp("window_height", w_height),
                  ::detail::make_nvp("out_channels", out_ch),
                  ::detail::make_nvp("connection_table", tbl),
                  ::detail::make_nvp("pad_type", pad_type),
                  ::detail::make_nvp("has_bias", has_bias),
                  ::detail::make_nvp("w_stride", w_stride),
                  ::detail::make_nvp("h_stride", h_stride));

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
    size_t in_dim, out_dim;
    bool has_bias;

    ::detail::arc(ar, ::detail::make_nvp("in_size", in_dim),
                  ::detail::make_nvp("out_size", out_dim),
                  ::detail::make_nvp("has_bias", has_bias));
    construct(in_dim, out_dim, has_bias);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::recurrent_cell_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar, cereal::construct<tiny_dnn::recurrent_cell_layer> &construct) {
    size_t in_dim, out_dim;
    bool has_bias;

    ::detail::arc(ar, ::detail::make_nvp("in_size", in_dim),
                  ::detail::make_nvp("out_size", out_dim),
                  ::detail::make_nvp("has_bias", has_bias));
    auto al = tiny_dnn::layer::load_layer(ar);
    // a nullptr is passed to avoid creating unused activation layer.
    construct(in_dim, out_dim, has_bias, nullptr);
    // set the activation to the loaded value
    construct->set_activation(
      std::static_pointer_cast<tiny_dnn::activation_layer>(al));
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::slice_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar, cereal::construct<tiny_dnn::slice_layer> &construct) {
    tiny_dnn::shape3d in_shape;
    tiny_dnn::slice_type slice_type;
    size_t num_outputs;

    ::detail::arc(ar, ::detail::make_nvp("in_size", in_shape),
                  ::detail::make_nvp("slice_type", slice_type),
                  ::detail::make_nvp("num_outputs", num_outputs));
    construct(in_shape, slice_type, num_outputs);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::sigmoid_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar, cereal::construct<tiny_dnn::sigmoid_layer> &construct) {
    tiny_dnn::shape3d in_shape;

    ::detail::arc(ar, ::detail::make_nvp("in_size", in_shape));
    construct(in_shape);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::tanh_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar, cereal::construct<tiny_dnn::tanh_layer> &construct) {
    tiny_dnn::shape3d in_shape;

    ::detail::arc(ar, ::detail::make_nvp("in_size", in_shape));
    construct(in_shape);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::relu_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar, cereal::construct<tiny_dnn::relu_layer> &construct) {
    tiny_dnn::shape3d in_shape;

    ::detail::arc(ar, ::detail::make_nvp("in_size", in_shape));
    construct(in_shape);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::softmax_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar, cereal::construct<tiny_dnn::softmax_layer> &construct) {
    tiny_dnn::shape3d in_shape;

    ::detail::arc(ar, ::detail::make_nvp("in_size", in_shape));
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

    ::detail::arc(ar, ::detail::make_nvp("in_size", in_shape),
                  ::detail::make_nvp("epsilon", epsilon));

    construct(in_shape, epsilon);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::selu_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar, cereal::construct<tiny_dnn::selu_layer> &construct) {
    tiny_dnn::shape3d in_shape;
    tiny_dnn::float_t lambda;
    tiny_dnn::float_t alpha;

    ::detail::arc(ar, ::detail::make_nvp("in_size", in_shape),
                  ::detail::make_nvp("lambda", lambda),
                  ::detail::make_nvp("alpha", alpha));
    construct(in_shape, lambda, alpha);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::elu_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar, cereal::construct<tiny_dnn::elu_layer> &construct) {
    tiny_dnn::shape3d in_shape;
    ::detail::arc(ar, ::detail::make_nvp("in_size", in_shape));
    construct(in_shape);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::tanh_p1m2_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar, cereal::construct<tiny_dnn::tanh_p1m2_layer> &construct) {
    tiny_dnn::shape3d in_shape;

    ::detail::arc(ar, ::detail::make_nvp("in_size", in_shape));
    construct(in_shape);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::softplus_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar, cereal::construct<tiny_dnn::softplus_layer> &construct) {
    tiny_dnn::shape3d in_shape;
    tiny_dnn::float_t beta;
    tiny_dnn::float_t threshold;

    ::detail::arc(ar, ::detail::make_nvp("in_size", in_shape),
                  ::detail::make_nvp("beta", beta),
                  ::detail::make_nvp("threshold", threshold));
    construct(in_shape, beta, threshold);
  }
};

template <>
struct LoadAndConstruct<tiny_dnn::softsign_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar, cereal::construct<tiny_dnn::softsign_layer> &construct) {
    tiny_dnn::shape3d in_shape;

    ::detail::arc(ar, ::detail::make_nvp("in_size", in_shape));
    construct(in_shape);
  }
};

}  // namespace cereal

namespace tiny_dnn {

struct serialization_buddy {
#ifndef CNN_NO_SERIALIZATION

  template <class Archive>
  static inline void serialize(Archive &ar, tiny_dnn::layer &layer) {
    auto all_weights = layer.weights();
    for (auto weight : all_weights) {
      ar(*weight);
    }
    layer.initialized_ = true;
  }

  template <class Archive>
  static inline void serialize(Archive &ar,
                               tiny_dnn::elementwise_add_layer &layer) {
    ::detail::arc(ar, ::detail::make_nvp("num_args", layer.num_args_),
                  ::detail::make_nvp("dim", layer.dim_));
  }

  template <class Archive>
  static inline void serialize(Archive &ar,
                               tiny_dnn::average_pooling_layer &layer) {
    ::detail::arc(ar, ::detail::make_nvp("in_size", layer.in_),
                  ::detail::make_nvp("pool_size_x", layer.pool_size_x_),
                  ::detail::make_nvp("pool_size_y", layer.pool_size_y_),
                  ::detail::make_nvp("stride_x", layer.stride_x_),
                  ::detail::make_nvp("stride_y", layer.stride_y_),
                  ::detail::make_nvp("pad_type", layer.pad_type_));
  }

  template <class Archive>
  static inline void serialize(Archive &ar,
                               tiny_dnn::average_unpooling_layer &layer) {
    ::detail::arc(ar, ::detail::make_nvp("in_size", layer.in_),
                  ::detail::make_nvp("pool_size", layer.w_.width_),
                  ::detail::make_nvp("stride", layer.stride_));
  }

  template <class Archive>
  static inline void serialize(Archive &ar,
                               tiny_dnn::batch_normalization_layer &layer) {
    ::detail::arc(ar,
                  ::detail::make_nvp("in_spatial_size", layer.in_spatial_size_),
                  ::detail::make_nvp("in_channels", layer.in_channels_),
                  ::detail::make_nvp("epsilon", layer.eps_),
                  ::detail::make_nvp("momentum", layer.momentum_),
                  ::detail::make_nvp("phase", layer.phase_),
                  ::detail::make_nvp("mean", layer.mean_),
                  ::detail::make_nvp("variance", layer.variance_));
  }

  template <class Archive>
  static inline void serialize(Archive &ar, tiny_dnn::concat_layer &layer) {
    ::detail::arc(ar, ::detail::make_nvp("in_size", layer.in_shapes_));
  }

  template <class Archive>
  static inline void serialize(Archive &ar,
                               tiny_dnn::convolutional_layer &layer) {
    auto &params_ = layer.params_;
    ::detail::arc(ar, ::detail::make_nvp("in_size", params_.in),
                  ::detail::make_nvp("window_width", params_.weight.width_),
                  ::detail::make_nvp("window_height", params_.weight.height_),
                  ::detail::make_nvp("out_channels", params_.out.depth_),
                  ::detail::make_nvp("connection_table", params_.tbl),
                  ::detail::make_nvp("pad_type", params_.pad_type),
                  ::detail::make_nvp("has_bias", params_.has_bias),
                  ::detail::make_nvp("w_stride", params_.w_stride),
                  ::detail::make_nvp("h_stride", params_.h_stride));
  }

  template <class Archive>
  static inline void serialize(Archive &ar,
                               tiny_dnn::deconvolutional_layer &layer) {
    auto &params_ = layer.params_;
    ::detail::arc(ar, ::detail::make_nvp("in_size", params_.in),
                  ::detail::make_nvp("window_width", params_.weight.width_),
                  ::detail::make_nvp("window_height", params_.weight.height_),
                  ::detail::make_nvp("out_channels", params_.out.depth_),
                  ::detail::make_nvp("connection_table", params_.tbl),
                  ::detail::make_nvp("pad_type", params_.pad_type),
                  ::detail::make_nvp("has_bias", params_.has_bias),
                  ::detail::make_nvp("w_stride", params_.w_stride),
                  ::detail::make_nvp("h_stride", params_.h_stride));
  }

  template <class Archive>
  static inline void serialize(Archive &ar, tiny_dnn::dropout_layer &layer) {
    ::detail::arc(ar, ::detail::make_nvp("in_size", layer.in_size_),
                  ::detail::make_nvp("dropout_rate", layer.dropout_rate_),
                  ::detail::make_nvp("phase", layer.phase_));
  }

  template <class Archive>
  static inline void serialize(Archive &ar,
                               tiny_dnn::fully_connected_layer &layer) {
    auto &params_ = layer.params_;
    ::detail::arc(ar, ::detail::make_nvp("in_size", params_.in_size_),
                  ::detail::make_nvp("out_size", params_.out_size_),
                  ::detail::make_nvp("has_bias", params_.has_bias_));
  }

  template <class Archive>
  static inline void serialize(Archive &ar,
                               tiny_dnn::global_average_pooling_layer &layer) {
    auto &params_ = layer.params_;
    ::detail::arc(ar, ::detail::make_nvp("in_shape", params_.in));
  }

  template <class Archive>
  static inline void serialize(Archive &ar, tiny_dnn::input_layer &layer) {
    ::detail::arc(ar, ::detail::make_nvp("shape", layer.shape_));
  }

  template <class Archive>
  static inline void serialize(Archive &ar, tiny_dnn::linear_layer &layer) {
    ::detail::arc(ar, ::detail::make_nvp("in_size", layer.dim_),
                  ::detail::make_nvp("scale", layer.scale_),
                  ::detail::make_nvp("bias", layer.bias_));
  }

  template <class Archive>
  static inline void serialize(Archive &ar, tiny_dnn::lrn_layer &layer) {
    ::detail::arc(ar, ::detail::make_nvp("in_shape", layer.in_shape_),
                  ::detail::make_nvp("size", layer.size_),
                  ::detail::make_nvp("alpha", layer.alpha_),
                  ::detail::make_nvp("beta", layer.beta_),
                  ::detail::make_nvp("region", layer.region_));
  }

  template <class Archive>
  static inline void serialize(Archive &ar,
                               tiny_dnn::max_pooling_layer &layer) {
    auto &params_ = layer.params_;
    ::detail::arc(ar, ::detail::make_nvp("in_size", params_.in),
                  ::detail::make_nvp("pool_size_x", params_.pool_size_x),
                  ::detail::make_nvp("pool_size_y", params_.pool_size_y),
                  ::detail::make_nvp("stride_x", params_.stride_x),
                  ::detail::make_nvp("stride_y", params_.stride_y),
                  ::detail::make_nvp("pad_type", params_.pad_type));
  }

  template <class Archive>
  static inline void serialize(Archive &ar,
                               tiny_dnn::max_unpooling_layer &layer) {
    ::detail::arc(ar, ::detail::make_nvp("in_size", layer.in_),
                  ::detail::make_nvp("unpool_size", layer.unpool_size_),
                  ::detail::make_nvp("stride", layer.stride_));
  }

  template <class Archive>
  static inline void serialize(Archive &ar, tiny_dnn::power_layer &layer) {
    ::detail::arc(ar, ::detail::make_nvp("in_size", layer.in_shape_),
                  ::detail::make_nvp("factor", layer.factor_),
                  ::detail::make_nvp("scale", layer.scale_));
  }

  template <class Archive>
  static inline void serialize(Archive &ar,
                               tiny_dnn::quantized_convolutional_layer &layer) {
    auto &params_ = layer.params_;
    ::detail::arc(ar, ::detail::make_nvp("in_size", params_.in),
                  ::detail::make_nvp("window_width", params_.weight.width_),
                  ::detail::make_nvp("window_height", params_.weight.height_),
                  ::detail::make_nvp("out_channels", params_.out.depth_),
                  ::detail::make_nvp("connection_table", params_.tbl),
                  ::detail::make_nvp("pad_type", params_.pad_type),
                  ::detail::make_nvp("has_bias", params_.has_bias),
                  ::detail::make_nvp("w_stride", params_.w_stride),
                  ::detail::make_nvp("h_stride", params_.h_stride));
  }

  template <class Archive>
  static inline void serialize(
    Archive &ar, tiny_dnn::quantized_deconvolutional_layer &layer) {
    auto &params_ = layer.params_;
    ::detail::arc(ar, ::detail::make_nvp("in_size", params_.in),
                  ::detail::make_nvp("window_width", params_.weight.width_),
                  ::detail::make_nvp("window_height", params_.weight.height_),
                  ::detail::make_nvp("out_channels", params_.out.depth_),
                  ::detail::make_nvp("connection_table", params_.tbl),
                  ::detail::make_nvp("pad_type", params_.pad_type),
                  ::detail::make_nvp("has_bias", params_.has_bias),
                  ::detail::make_nvp("w_stride", params_.w_stride),
                  ::detail::make_nvp("h_stride", params_.h_stride));
  }

  template <class Archive>
  static inline void serialize(
    Archive &ar, tiny_dnn::quantized_fully_connected_layer &layer) {
    auto &params_ = layer.params_;
    ::detail::arc(ar, ::detail::make_nvp("in_size", params_.in_size_),
                  ::detail::make_nvp("out_size", params_.out_size_),
                  ::detail::make_nvp("has_bias", params_.has_bias_));
  }

  template <class Archive>
  static inline void serialize(Archive &ar,
                               tiny_dnn::recurrent_cell_layer &layer) {
    auto &params_ = layer.params_;
    ::detail::arc(ar, ::detail::make_nvp("in_size", params_.in_size_),
                  ::detail::make_nvp("out_size", params_.out_size_),
                  ::detail::make_nvp("has_bias", params_.has_bias_));
    tiny_dnn::layer::save_layer(ar, *params_.activation_);
  }

  template <class Archive>
  static inline void serialize(Archive &ar, tiny_dnn::slice_layer &layer) {
    ::detail::arc(ar, ::detail::make_nvp("in_size", layer.in_shape_),
                  ::detail::make_nvp("slice_type", layer.slice_type_),
                  ::detail::make_nvp("num_outputs", layer.num_outputs_));
  }

  template <class Archive>
  static inline void serialize(Archive &ar, tiny_dnn::sigmoid_layer &layer) {
    ::detail::arc(ar, ::detail::make_nvp("in_size", layer.in_shape()[0]));
  }

  template <class Archive>
  static inline void serialize(Archive &ar, tiny_dnn::tanh_layer &layer) {
    ::detail::arc(ar, ::detail::make_nvp("in_size", layer.in_shape()[0]));
  }

  template <class Archive>
  static inline void serialize(Archive &ar, tiny_dnn::relu_layer &layer) {
    ::detail::arc(ar, ::detail::make_nvp("in_size", layer.in_shape()[0]));
  }

  template <class Archive>
  static inline void serialize(Archive &ar, tiny_dnn::softmax_layer &layer) {
    ::detail::arc(ar, ::detail::make_nvp("in_size", layer.in_shape()[0]));
  }

  template <class Archive>
  static inline void serialize(Archive &ar, tiny_dnn::leaky_relu_layer &layer) {
    ::detail::arc(ar, ::detail::make_nvp("in_size", layer.in_shape()[0]),
                  ::detail::make_nvp("epsilon", layer.epsilon_));
  }

  template <class Archive>
  static inline void serialize(Archive &ar, tiny_dnn::elu_layer &layer) {
    tiny_dnn::shape3d shape = layer.in_shape()[0];
    ::detail::arc(ar, ::detail::make_nvp("in_size", shape));
  }

  template <class Archive>
  static inline void serialize(Archive &ar, tiny_dnn::selu_layer &layer) {
    ::detail::arc(ar, ::detail::make_nvp("in_size", layer.in_shape()[0]),
                  ::detail::make_nvp("lambda", layer.lambda_),
                  ::detail::make_nvp("alpha", layer.alpha_));
  }

  template <class Archive>
  static inline void serialize(Archive &ar, tiny_dnn::tanh_p1m2_layer &layer) {
    ::detail::arc(ar, ::detail::make_nvp("in_size", layer.in_shape()[0]));
  }

  template <class Archive>
  static inline void serialize(Archive &ar, tiny_dnn::softplus_layer &layer) {
    ::detail::arc(ar, ::detail::make_nvp("in_size", layer.in_shape()[0]),
                  ::detail::make_nvp("beta", layer.beta_),
                  ::detail::make_nvp("threshold", layer.threshold_));
  }

  template <class Archive>
  static inline void serialize(Archive &ar, tiny_dnn::softsign_layer &layer) {
    ::detail::arc(ar, ::detail::make_nvp("in_size", layer.in_shape()[0]));
  }

#endif  // #ifndef CNN_NO_SERIALIZATION
};      // struct serialization_buddy

template <class Archive, typename T>
typename std::enable_if<std::is_base_of<tiny_dnn::layer, T>::value>::type
serialize(Archive &ar, T &layer) {
  auto &inconstant_layer =
    const_cast<typename std::remove_const<T>::type &>(layer);
  inconstant_layer.serialize_prolog(ar);
  serialization_buddy::serialize(ar, inconstant_layer);
}

template <class Archive, typename T>
void serialize(Archive &ar, tiny_dnn::index3d<T> &idx) {
  ::detail::arc(ar, ::detail::make_nvp("width", idx.width_),
                ::detail::make_nvp("height", idx.height_),
                ::detail::make_nvp("depth", idx.depth_));
}

namespace core {

template <class Archive>
void serialize(Archive &ar, tiny_dnn::core::connection_table &tbl) {
  ::detail::arc(ar, ::detail::make_nvp("rows", tbl.rows_),
                ::detail::make_nvp("cols", tbl.cols_));
  if (tbl.is_empty()) {
    std::string all("all");
    ::detail::arc(ar, ::detail::make_nvp("connection", all));
  } else {
    ::detail::arc(ar, ::detail::make_nvp("connection", tbl.connected_));
  }
}

}  // namespace core

}  // namespace tiny_dnn
