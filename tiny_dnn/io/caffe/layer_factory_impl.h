/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tiny_dnn/io/caffe/caffe.pb.h"

#include "tiny_dnn/layers/average_pooling_layer.h"
#include "tiny_dnn/layers/convolutional_layer.h"
#include "tiny_dnn/layers/deconvolutional_layer.h"
#include "tiny_dnn/layers/dropout_layer.h"
#include "tiny_dnn/layers/fully_connected_layer.h"
#include "tiny_dnn/layers/linear_layer.h"
#include "tiny_dnn/layers/lrn_layer.h"
#include "tiny_dnn/layers/max_pooling_layer.h"

typedef tiny_dnn::shape3d shape_t;

#if defined(_MSC_VER) || defined(WIN32)
#define _NOMINMAX
#include <fcntl.h>
#include <io.h>
#define CNN_OPEN_BINARY(filename) open(filename, _O_RDONLY | _O_BINARY)
#define CNN_OPEN_TXT(filename) open(filename, _O_RDONLY)
#pragma warning(push)
#pragma warning(disable : 4996)
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#define CNN_OPEN_BINARY(filename) open(filename, O_RDONLY)
#define CNN_OPEN_TXT(filename) open(filename, O_RDONLY)
#endif

namespace tiny_dnn {
namespace detail {

inline void read_proto_from_text(const std::string &prototxt,
                                 google::protobuf::Message *message) {
  int fd = CNN_OPEN_TXT(prototxt.c_str());
  if (fd == -1) {
    throw nn_error("file not found: " + prototxt);
  }

  google::protobuf::io::FileInputStream input(fd);
  input.SetCloseOnDelete(true);

  if (!google::protobuf::TextFormat::Parse(&input, message)) {
    throw nn_error("failed to parse");
  }
}

inline void read_proto_from_binary(const std::string &protobinary,
                                   google::protobuf::Message *message) {
  int fd = CNN_OPEN_BINARY(protobinary.c_str());
  google::protobuf::io::FileInputStream rawstr(fd);
  google::protobuf::io::CodedInputStream codedstr(&rawstr);

  rawstr.SetCloseOnDelete(true);
  codedstr.SetTotalBytesLimit(std::numeric_limits<int>::max(),
                              std::numeric_limits<int>::max() / 2);

  if (!message->ParseFromCodedStream(&codedstr)) {
    throw nn_error("failed to parse");
  }
}

inline std::shared_ptr<weight_init::function> create_filler(
  const std::string &filler) {
  if (filler == "xavier") {
    return std::make_shared<weight_init::xavier>();
  } else if (filler == "constant") {
    return std::make_shared<weight_init::constant>();
  } else if (filler == "gaussian") {
    return std::make_shared<weight_init::gaussian>();
  } else {
    throw nn_error("unsupported filler type");
  }
}

template <typename param>
inline bool get_kernel_size_2d(const param &p, layer_size_t *kernel) {
  if (p.has_kernel_w() && p.has_kernel_h()) {
    if (p.kernel_w() != p.kernel_h()) {
      throw nn_error("unsupported kernel shape");
    }
    *kernel = p.kernel_w();
    return true;
  }
  return false;
}

template <typename param>
inline bool get_kernel_size_2d(const param &p,
                               layer_size_t *kernel_w,
                               layer_size_t *kernel_h) {
  if (p.has_kernel_w() && p.has_kernel_h()) {
    *kernel_w = p.kernel_w();
    *kernel_h = p.kernel_h();
    return true;
  }
  return false;
}

inline layer_size_t get_kernel_size_2d(const caffe::ConvolutionParameter &p) {
  layer_size_t window_size;
  if (!get_kernel_size_2d(p, &window_size)) {
    if (p.kernel_size_size() > 1) {
      throw nn_error("unsupported kernel shape");
    }
    window_size = p.kernel_size(0);
  }
  return window_size;
}

inline std::shared_ptr<layer> create_max_pool(layer_size_t pool_size_w,
                                              layer_size_t pool_size_h,
                                              layer_size_t stride_w,
                                              layer_size_t stride_h,
                                              bool ceil_mode,
                                              padding pad_type,
                                              const shape_t &bottom_shape,
                                              shape_t *top_shape) {
  using max_pool = max_pooling_layer;
  auto mp        = std::make_shared<max_pool>(
    bottom_shape.width_, bottom_shape.height_, bottom_shape.depth_, pool_size_w,
    pool_size_h, stride_w, stride_h, ceil_mode, pad_type);

  *top_shape = mp->out_shape()[0];
  mp->init_weight();

  return mp;
}

inline std::shared_ptr<layer> create_ave_pool(layer_size_t pool_size_w,
                                              layer_size_t pool_size_h,
                                              layer_size_t stride_w,
                                              layer_size_t stride_h,
                                              bool ceil_mode,
                                              padding pad_type,
                                              const shape_t &bottom_shape,
                                              shape_t *top_shape) {
  using ave_pool = average_pooling_layer;
  auto ap        = std::make_shared<ave_pool>(
    bottom_shape.width_, bottom_shape.height_, bottom_shape.depth_, pool_size_w,
    pool_size_h, stride_w, stride_h, ceil_mode, pad_type);

  // tiny-dnn has trainable parameter in average-pooling layer
  float_t weight = 1.0 / (pool_size_w * pool_size_h);

  vec_t &w = *ap->weights()[0];
  vec_t &b = *ap->weights()[1];

  vectorize::fill(&w[0], w.size(), weight);
  vectorize::fill(&b[0], b.size(), float_t{0});

  *top_shape = ap->out_shape()[0];
  ap->init_weight();
  ap->set_trainable(false);

  return ap;
}

inline std::shared_ptr<layer> create_softmax(const caffe::LayerParameter &layer,
                                             const shape_t &bottom_shape,
                                             shape_t *) {
  auto sm = std::make_shared<softmax_layer>(bottom_shape.size());
  sm->init_weight();
  return sm;
}

inline std::shared_ptr<layer> create_sigmoid(const caffe::LayerParameter &layer,
                                             const shape_t &bottom_shape,
                                             shape_t *) {
  auto ce = std::make_shared<sigmoid_layer>(bottom_shape.size());
  return ce;
}

inline std::shared_ptr<layer> create_tanh(const caffe::LayerParameter &layer,
                                          const shape_t &bottom_shape,
                                          shape_t *) {
  auto tanh = std::make_shared<tanh_layer>(bottom_shape.size());
  return tanh;
}

inline std::shared_ptr<layer> create_power(const caffe::LayerParameter &layer,
                                           const shape_t &bottom_shape,
                                           shape_t *) {
  auto power = std::make_shared<power_layer>(
    bottom_shape, layer.power_param().power(), layer.power_param().scale());
  return power;
}

inline std::shared_ptr<layer> create_pooling(const caffe::LayerParameter &layer,
                                             const shape_t &bottom_shape,
                                             shape_t *top_shape) {
  if (!layer.has_pooling_param()) {
    throw nn_error("pool param missing");
  }

  auto pool_param = layer.pooling_param();

  layer_size_t h_stride    = 0;
  layer_size_t w_stride    = 0;
  layer_size_t pool_size_w = 0;
  layer_size_t pool_size_h = 0;
  layer_size_t h_pad       = 0;
  layer_size_t w_pad       = 0;
  bool ceil_mode           = false;
  padding pad_type         = padding::valid;

  if (!get_kernel_size_2d(pool_param, &pool_size_w, &pool_size_h)) {
    pool_size_w = pool_size_h = pool_param.kernel_size();
  }

  if (pool_param.has_stride() || pool_param.has_stride_h()) {
    h_stride =
      pool_param.has_stride() ? pool_param.stride() : pool_param.stride_h();
  }

  if (pool_param.has_stride() || pool_param.has_stride_w()) {
    w_stride =
      pool_param.has_stride() ? pool_param.stride() : pool_param.stride_w();
  }

  if (pool_param.has_pad() || pool_param.has_pad_w()) {
    w_pad = pool_param.has_pad() ? pool_param.pad() : pool_param.pad_w();
  }

  if (pool_param.has_pad() || pool_param.has_pad_h()) {
    h_pad = pool_param.has_pad() ? pool_param.pad() : pool_param.pad_h();
  }

  if (w_pad != 0) {
    if (w_pad == (pool_size_w - 1) / 2) {
      pad_type = padding::same;
    } else {
      throw nn_error("unsupported padding type");
    }
    // NB: w_pad == pool_size_w - 1 could lead to pad_type = padding::full,
    //     if such a type existed
  }

  if (h_pad != 0) {
    if (h_pad == (pool_size_h - 1) / 2) {
      pad_type = padding::same;
    } else {
      throw nn_error("unsupported padding type");
    }
    // NB: h_pad == pool_size_h - 1 could lead to pad_type = padding::full,
    //     if such a type existed
  }

  if (pool_param.has_ceil_mode()) {
    ceil_mode = pool_param.ceil_mode();
  }

  if (pool_param.has_pool()) {
    auto type = pool_param.pool();

    switch (type) {
      case caffe::PoolingParameter_PoolMethod_MAX:
        return create_max_pool(pool_size_w, pool_size_h, w_stride, h_stride,
                               ceil_mode, pad_type, bottom_shape, top_shape);
      case caffe::PoolingParameter_PoolMethod_AVE:
        return create_ave_pool(pool_size_w, pool_size_h, w_stride, h_stride,
                               ceil_mode, pad_type, bottom_shape, top_shape);
      default: throw nn_error("unsupported layer type");
    }
  }

  // default: max-pool
  return create_max_pool(pool_size_w, pool_size_h, w_stride, h_stride, ceil_mode,
                         pad_type, bottom_shape, top_shape);
}

inline std::shared_ptr<layer> create_relu(const caffe::LayerParameter &layer,
                                          const shape_t &bottom_shape,
                                          shape_t *) {
  auto relu = std::make_shared<relu_layer>(bottom_shape.size());
  return relu;
}

inline std::shared_ptr<layer> create_elu(const caffe::LayerParameter &layer,
                                         const shape_t &bottom_shape,
                                         shape_t *) {
  auto elu = std::make_shared<elu_layer>(bottom_shape.size());
  return elu;
}

inline std::shared_ptr<layer> create_batchnorm(
  const caffe::LayerParameter &layer,
  const shape_t &bottom_shape,
  shape_t *top_shape) {
  using bn_layer = batch_normalization_layer;

  *top_shape = bottom_shape;

  float_t eps      = 1e-5f;
  float_t momentum = 0.999f;

  if (layer.has_batch_norm_param()) {
    auto bn_param = layer.batch_norm_param();

    if (bn_param.has_eps()) {
      eps = bn_param.eps();
    }
    if (bn_param.has_moving_average_fraction()) {
      momentum = bn_param.moving_average_fraction();
    }
  }

  auto bn = std::make_shared<bn_layer>(bottom_shape.area(), bottom_shape.depth_,
                                       eps, momentum, net_phase::test);

  // weight
  if (layer.blobs_size() > 0) {
    auto global_stats = layer.blobs();
    if (global_stats.size() != 3) {
      throw std::runtime_error("unexpected bn stored statistics");
    }

    float_t scale_factor =
      global_stats.Get(2).data(0) == 0 ? 0 : 1 / global_stats.Get(2).data(0);
    vec_t mean(bottom_shape.depth_);
    vec_t variance(bottom_shape.depth_);

    for (size_t i = 0; i < mean.size(); i++) {
      mean[i]     = global_stats.Get(0).data(i) * scale_factor;
      variance[i] = global_stats.Get(1).data(i) * scale_factor;
    }
    bn->set_mean(mean);
    bn->set_variance(variance);
  }

  return bn;
}

inline void load_weights_fullyconnected(const caffe::LayerParameter &src,
                                        layer *dst) {
  auto weights = src.blobs(0);
  int curr     = 0;

  const auto dst_out_size = dst->out_size();
  const auto dst_in_size  = dst->in_size();

  if (dst_out_size * dst_in_size != static_cast<size_t>(weights.data_size())) {
    throw nn_error(std::string("layer size mismatch!") + "caffe(" + src.name() +
                   "):" + to_string(weights.data_size()) + "\n" + "tiny-dnn(" +
                   dst->layer_type() + "):" + to_string(dst->weights().size()));
  }

  vec_t &w = *dst->weights()[0];
  vec_t &b = *dst->weights()[1];

  // fill weights
  for (size_t o = 0; o < dst_out_size; o++) {
    for (size_t i = 0; i < dst_in_size; i++) {
      // TODO(karandesai): how to access to weights?
      // dst->weight()[i * dst->out_size() + o] = weights.data(curr++); //
      // transpose
      w[i * dst_out_size + o] = weights.data(curr++);  // transpose
    }
  }

  // fill bias
  if (src.inner_product_param().bias_term()) {
    auto biases = src.blobs(1);
    for (size_t o = 0; o < dst_out_size; o++) {
      // TODO(karandesai): how to access to biases?
      // dst->bias()[o] = biases.data(o);
      b[o] = biases.data(o);
    }
  }
}

inline std::shared_ptr<layer> create_fullyconnected(
  const caffe::LayerParameter &layer,
  const shape_t &bottom_shape,
  shape_t *top_shape) {
  using fc_layer = fully_connected_layer;

  if (!layer.has_inner_product_param()) {
    throw nn_error("inner-product param missing");
  }

  layer_size_t dim_input = 0, dim_output = 0;
  bool has_bias = true;

  auto ip_param = layer.inner_product_param();
  has_bias      = ip_param.bias_term();

  dim_output = ip_param.num_output();
  dim_input  = bottom_shape.size();

  auto ip = std::make_shared<fc_layer>(dim_input, dim_output, has_bias);

  // filler
  if (ip_param.has_weight_filler()) {
    ip->weight_init(create_filler(ip_param.weight_filler().type()));
  }

  if (ip_param.has_bias_filler()) {
    ip->bias_init(create_filler(ip_param.bias_filler().type()));
  }

  // weight
  if (layer.blobs_size() > 0) {
    load_weights_fullyconnected(layer, ip.get());
  }

  // TODO(karan): check if it works
  *top_shape = ip->out_shape()[0];
  return ip;
}

inline void load_weights_conv(const caffe::LayerParameter &src, layer *dst) {
  // fill weight
  auto weights = src.blobs(0);

  // TODO(karan): check if it works
  // int out_channels = dst->out_shape().depth_;
  // int in_channels = dst->in_shape().depth_;
  int out_channels = dst->out_data_shape()[0].depth_;
  int in_channels  = dst->in_data_shape()[0].depth_;

  core::connection_table table;
  auto conv_param = src.convolution_param();
  int dst_idx     = 0;
  int src_idx     = 0;
  int window_size = get_kernel_size_2d(conv_param);

  if (conv_param.has_group()) {
    table =
      core::connection_table(conv_param.group(), in_channels, out_channels);
  }

  vec_t &w = *dst->weights()[0];
  vec_t &b = *dst->weights()[1];

  // fill weights
  for (int o = 0; o < out_channels; o++) {
    for (int i = 0; i < in_channels; i++) {
      if (!table.is_connected(o, i)) {
        dst_idx += window_size * window_size;
        continue;
      }
      for (int x = 0; x < window_size * window_size; x++) {
        // dst->weight()[dst_idx++] = weights.data(src_idx++);
        w[dst_idx++] = weights.data(src_idx++);
      }
    }
  }

  // fill bias
  if (conv_param.bias_term()) {
    auto biases = src.blobs(1);
    for (int o = 0; o < out_channels; o++) {
      // dst->bias()[o] = biases.data(o);
      b[o] = biases.data(o);
    }
  }
}

inline void load_weights_batchnorm(const caffe::LayerParameter &src,
                                   layer *dst) {
  if (dst->layer_type() != "batch-norm")
    throw nn_error("batch-norm layer expected");

  if (src.blobs_size() > 0) {
    auto global_stats = src.blobs();
    if (global_stats.size() != 3) {
      throw nn_error("unexpected format for batch-norm statistics");
    }

    float_t scale_factor =
      global_stats.Get(2).data(0) == 0 ? 0 : 1 / global_stats.Get(2).data(0);

    int in_channels = dst->in_shape().at(0).depth_;
    vec_t mean(in_channels);
    vec_t variance(in_channels);

    for (size_t i = 0; i < mean.size(); i++) {
      mean[i]     = global_stats.Get(0).data(i) * scale_factor;
      variance[i] = global_stats.Get(1).data(i) * scale_factor;
    }
    auto bnl = dynamic_cast<batch_normalization_layer *>(dst);
    bnl->set_mean(mean);
    bnl->set_variance(variance);

  } else {
    throw nn_error("batch-norm layer missing blobs");
  }

  return;
}

inline void load_weights_pool(const caffe::LayerParameter &src, layer *dst) {
  auto pool_param = src.pooling_param();

  if (dst->weights().size()) {
    layer_size_t pool_size = 0;

    if (!get_kernel_size_2d(pool_param, &pool_size)) {
      pool_size = pool_param.kernel_size();
    }

    // tiny-dnn has trainable parameter in average-pooling layer
    float_t weight = 1.0 / sqr(pool_size);

    // TODO(karan)
    /*if (!dst->weight().empty()) {
        std::fill(dst->weight().begin(), dst->weight().end(), weight);
    }
    if (!dst->bias().empty()) {
        std::fill(dst->bias().begin(), dst->bias().end(), float_t{0});
        dst->init_bias();
    }*/

    vec_t &w = *dst->weights()[0];
    vec_t &b = *dst->weights()[1];

    if (!w.empty()) {
      vectorize::fill(&w[0], w.size(), weight);
    }
    if (!b.empty()) {
      vectorize::fill(&b[0], b.size(), float_t{0});
      // dst->init_bias();
    }
  }
}

inline std::shared_ptr<layer> create_lrn(const caffe::LayerParameter &layer,
                                         const shape_t &bottom_shape,
                                         shape_t *top_shape) {
  if (!layer.has_lrn_param()) {
    throw nn_error("lrn param missing");
  }

  auto lrn_param          = layer.lrn_param();
  layer_size_t local_size = 5;
  float_t alpha           = 1;
  float_t beta            = 5;
  norm_region region      = norm_region::across_channels;

  if (lrn_param.has_local_size()) local_size = lrn_param.local_size();
  if (lrn_param.has_alpha()) alpha           = lrn_param.alpha();
  if (lrn_param.has_beta()) beta             = lrn_param.beta();
  if (lrn_param.has_norm_region()) {
    if (lrn_param.norm_region() ==
        caffe::LRNParameter_NormRegion_WITHIN_CHANNEL)  // NOLINT
      region = norm_region::within_channels;
  }

  auto lrn = std::make_shared<lrn_layer>(
    bottom_shape.width_, bottom_shape.height_, local_size, bottom_shape.depth_,
    alpha, beta, region);
  return lrn;
}

inline std::shared_ptr<layer> create_dropout(const caffe::LayerParameter &layer,
                                             const shape_t &bottom_shape,
                                             shape_t *top_shape) {
  if (!layer.has_dropout_param()) {
    throw nn_error("dropout param missing");
  }

  float_t dropout_rate = float_t(0.5);

  if (layer.dropout_param().has_dropout_ratio()) {
    dropout_rate = layer.dropout_param().dropout_ratio();
  }

  auto dropout = std::make_shared<dropout_layer>(bottom_shape.size(),
                                                 dropout_rate, net_phase::test);
  return dropout;
}

inline std::shared_ptr<layer> create_convlayer(
  const caffe::LayerParameter &layer,
  const shape_t &bottom_shape,
  shape_t *top_shape) {
  using conv_layer = convolutional_layer;

  if (!layer.has_convolution_param()) {
    throw nn_error("convolution param missing");
  }

  // layer parameters
  layer_size_t in_width = 0, in_height = 0, window_size = 0;
  layer_size_t in_channels = 0, out_channels = 0;
  layer_size_t w_stride = 1, h_stride = 1;
  bool has_bias    = true;
  padding pad_type = padding::valid;
  core::connection_table table;

  auto conv_param = layer.convolution_param();

  // shape
  out_channels = conv_param.num_output();
  in_channels  = bottom_shape.depth_;
  in_width     = bottom_shape.width_;
  in_height    = bottom_shape.height_;
  has_bias     = conv_param.bias_term();
  window_size  = get_kernel_size_2d(conv_param);

  // padding
  if (conv_param.pad_size() == 1 ||
      (conv_param.has_pad_w() && conv_param.has_pad_h())) {
    uint32_t pad_w =
      conv_param.pad_size() == 1 ? conv_param.pad(0) : conv_param.pad_w();

    uint32_t pad_h =
      conv_param.pad_size() == 1 ? conv_param.pad(0) : conv_param.pad_h();

    if (pad_w != pad_h) {
      throw nn_error("conv:not supported padding size");
    }

    // 0 ... valid, (window_size-1)/2 ... same
    if (pad_w == (window_size - 1) / 2) {
      pad_type = padding::same;
    } else if (pad_w == 0) {
      pad_type = padding::valid;
    } else {
      throw nn_error("conv:not supported padding size");
    }
  }

  // stride
  if (conv_param.stride_size() == 1 || conv_param.has_stride_h()) {
    h_stride = conv_param.stride_size() == 1 ? conv_param.stride(0)
                                             : conv_param.stride_h();
  }

  if (conv_param.stride_size() == 1 || conv_param.has_stride_w()) {
    w_stride = conv_param.stride_size() == 1 ? conv_param.stride(0)
                                             : conv_param.stride_w();
  }

  // group
  if (conv_param.has_group()) {
    table =
      core::connection_table(conv_param.group(), in_channels, out_channels);
  }

  auto conv = std::make_shared<conv_layer>(
    in_width, in_height, window_size, in_channels, out_channels, table,
    pad_type, has_bias, w_stride, h_stride);
  // filler
  if (conv_param.has_weight_filler()) {
    conv->weight_init(create_filler(conv_param.weight_filler().type()));
  }

  if (conv_param.has_bias_filler()) {
    conv->bias_init(create_filler(conv_param.bias_filler().type()));
  }

  // set weight (optional)
  if (layer.blobs_size() > 0) {  // blobs(0)...weight, blobs(1)...bias
    load_weights_conv(layer, conv.get());
  }

  *top_shape = conv->out_shape()[0];
  return conv;
}

inline std::shared_ptr<layer> create_deconvlayer(
  const caffe::LayerParameter &layer,
  const shape_t &bottom_shape,
  shape_t *top_shape) {
  using deconv_layer = deconvolutional_layer;

  if (!layer.has_convolution_param()) {
    throw nn_error("deconvolution param missing");
  }

  // layer parameters
  layer_size_t in_width = 0, in_height = 0, window_size = 0;
  layer_size_t in_channels = 0, out_channels = 0;
  layer_size_t w_stride = 1, h_stride = 1;
  bool has_bias    = true;
  padding pad_type = padding::valid;
  core::connection_table table;

  auto deconv_param = layer.convolution_param();

  // shape
  out_channels = deconv_param.num_output();
  in_channels  = bottom_shape.depth_;
  in_width     = bottom_shape.width_;
  in_height    = bottom_shape.height_;
  has_bias     = deconv_param.bias_term();
  window_size  = get_kernel_size_2d(deconv_param);

  // unpadding
  if (deconv_param.pad_size() == 1 ||
      (deconv_param.has_pad_w() && deconv_param.has_pad_h())) {
    uint32_t unpad_w =
      deconv_param.pad_size() == 1 ? deconv_param.pad(0) : deconv_param.pad_w();

    uint32_t unpad_h =
      deconv_param.pad_size() == 1 ? deconv_param.pad(0) : deconv_param.pad_h();

    if (unpad_w != unpad_h) {
      throw nn_error("deconv:not supported unpadding size");
    }

    // 0 ... valid, (window_size-1)/2 ... same
    if (unpad_w == (window_size - 1) / 2) {
      pad_type = padding::same;
    } else if (unpad_w == 0) {
      pad_type = padding::valid;
    } else {
      throw nn_error("deconv:not supported unpadding size");
    }
  }

  // stride
  if (deconv_param.stride_size() == 1 || deconv_param.has_stride_h()) {
    h_stride = deconv_param.stride_size() == 1 ? deconv_param.stride(0)
                                               : deconv_param.stride_h();
  }

  if (deconv_param.stride_size() == 1 || deconv_param.has_stride_w()) {
    w_stride = deconv_param.stride_size() == 1 ? deconv_param.stride(0)
                                               : deconv_param.stride_w();
  }

  // group
  if (deconv_param.has_group()) {
    table =
      core::connection_table(deconv_param.group(), in_channels, out_channels);
  }

  auto deconv = std::make_shared<deconv_layer>(
    in_width, in_height, window_size, in_channels, out_channels, table,
    pad_type, has_bias, w_stride, h_stride);
  // filler
  if (deconv_param.has_weight_filler()) {
    deconv->weight_init(create_filler(deconv_param.weight_filler().type()));
  }

  if (deconv_param.has_bias_filler()) {
    deconv->bias_init(create_filler(deconv_param.bias_filler().type()));
  }

  // set weight (optional)
  if (layer.blobs_size() > 0) {  // blobs(0)...weight, blobs(1)...bias
    load_weights_conv(layer, deconv.get());
  }
  *top_shape = deconv->out_shape()[0];
  return deconv;
}

inline bool layer_skipped(const std::string &type) {
  if (type == "Data" || type == "EuclideanLoss" || type == "Input" ||
      type == "HDF5Data" || type == "Split" || type == "Accuracy")
    return true;
  return false;
}

inline bool layer_has_weights(const std::string &type) {
  static const char *activations[] = {"SoftmaxWithLoss",
                                      "SigmoidCrossEntropyLoss",
                                      "LRN",
                                      "Dropout",
                                      "ReLU",
                                      "ELU",
                                      "Sigmoid",
                                      "TanH",
                                      "Softmax"};
  for (unsigned int i = 0; i < sizeof(activations) / sizeof(activations[0]);
       i++) {
    if (activations[i] == type) return false;
  }
  return true;
}

inline bool layer_supported(const std::string &type) {
  static const char *supported[] = {"InnerProduct",
                                    "Convolution",
                                    "Deconvolution",
                                    "Pooling",
                                    "LRN",
                                    "Dropout",
                                    "SoftmaxWithLoss",
                                    "SigmoidCrossEntropyLoss",
                                    "ReLU",
                                    "ELU",
                                    "Sigmoid",
                                    "TanH",
                                    "Softmax",
                                    "BatchNorm",
                                    "Power"};

  for (size_t i = 0; i < sizeof(supported) / sizeof(supported[0]); i++) {
    if (supported[i] == type) return true;
  }
  return false;
}

inline bool layer_match(const std::string &caffetype,
                        const std::string &tiny_dnn_type) {
  const char *conversions[][2] = {{"InnerProduct", "fully-connected"},
                                  {"Convolution", "conv"},
                                  {"Deconvolution", "deconv"},
                                  {"Pooling", "ave-pool"},
                                  {"Pooling", "max-pool"},
                                  {"BatchNorm", "batch-norm"}};

  for (size_t i = 0; i < sizeof(conversions) / sizeof(conversions[0]); i++) {
    if (conversions[i][0] == caffetype && conversions[i][1] == tiny_dnn_type)
      return true;
  }
  return false;
}

inline std::shared_ptr<layer> create(const caffe::LayerParameter &layer,
                                     const shape_t &in_shape,
                                     shape_t *out_shape) {
  const std::string layer_type = layer.type();

  if (layer_type == "Convolution") {
    return detail::create_convlayer(layer, in_shape, out_shape);
  }

  if (layer_type == "Deconvolution") {
    return detail::create_deconvlayer(layer, in_shape, out_shape);
  }

  if (layer_type == "InnerProduct") {
    return detail::create_fullyconnected(layer, in_shape, out_shape);
  }

  if (layer_type == "Pooling") {
    return detail::create_pooling(layer, in_shape, out_shape);
  }

  if (layer_type == "BatchNorm") {
    return detail::create_batchnorm(layer, in_shape, out_shape);
  }

  if (layer_type == "LRN") {
    return detail::create_lrn(layer, in_shape, out_shape);
  }

  if (layer_type == "Dropout") {
    return detail::create_dropout(layer, in_shape, out_shape);
  }

  if (layer_type == "SoftmaxWithLoss" || layer_type == "Softmax") {
    return detail::create_softmax(layer, in_shape, out_shape);
  }

  if (layer_type == "SigmoidCrossEntropyLoss" || layer_type == "Sigmoid") {
    return detail::create_sigmoid(layer, in_shape, out_shape);
  }

  if (layer_type == "ReLU") {
    return detail::create_relu(layer, in_shape, out_shape);
  }

  if (layer_type == "ELU") {
    return detail::create_elu(layer, in_shape, out_shape);
  }

  if (layer_type == "TanH") {
    return detail::create_tanh(layer, in_shape, out_shape);
  }

  if (layer_type == "Power") {
    return detail::create_power(layer, in_shape, out_shape);
  }

  throw nn_error("layer parser not found");
}

inline void load(const caffe::LayerParameter &src, layer *dst) {
  typedef std::function<void(const caffe::LayerParameter &, layer *)>
    factoryimpl;  // NOLINT
  std::unordered_map<std::string, factoryimpl> factory_registry;

  factory_registry["Convolution"]   = detail::load_weights_conv;
  factory_registry["Deconvolution"] = detail::load_weights_conv;
  factory_registry["InnerProduct"]  = detail::load_weights_fullyconnected;
  factory_registry["Pooling"]       = detail::load_weights_pool;
  factory_registry["BatchNorm"]     = detail::load_weights_batchnorm;

  if (factory_registry.find(src.type()) == factory_registry.end()) {
    throw nn_error("layer parser not found");
  }

  return factory_registry[src.type()](src, dst);
}

struct layer_node {
  const caffe::LayerParameter *layer;
  const layer_node *next;  // top-side
  const layer_node *prev;  // bottom-side

  layer_node() : layer(0), next(0), prev(0) {}
  explicit layer_node(const caffe::LayerParameter *l)
    : layer(l), next(0), prev(0) {}
};

// parse caffe net and interpret as single layer vector
class caffe_layer_vector {
 public:
  explicit caffe_layer_vector(const caffe::NetParameter &net_orig)
    : net(net_orig) {
    if (net.layers_size() > 0) {
      upgradev1net(net_orig, &net);
    }

    nodes.reserve(net.layer_size());

    for (int i = 0; i < net.layer_size(); i++) {
      auto &l = net.layer(i);

      if (layer_table.find(l.name()) != layer_table.end()) continue;

      nodes.emplace_back(&l);
      layer_table[l.name()] = &nodes.back();
    }

    for (size_t i = 0; i < nodes.size(); i++) {
      auto &l = nodes[i];

      if (l.layer->bottom_size() > 0 && blob_table[l.layer->bottom(0)]) {
        auto &bottom = blob_table[l.layer->bottom(0)];
        l.prev       = bottom;
        layer_table[bottom->layer->name()]->next = &l;
      }

      if (l.layer->top_size() > 0) {
        blob_table[l.layer->top(0)] = &l;
      }
    }

    auto root = std::find_if(nodes.begin(), nodes.end(),
                             [](const layer_node &n) { return n.prev == 0; });

    if (root == nodes.end()) {
      throw nn_error("root layer not found");
    }

    root_node                 = &*root;
    const layer_node *current = &*root;

    while (current) {
      node_list.push_back(current->layer);
      current = current->next;
    }
  }

  size_t size() const { return node_list.size(); }

  const caffe::LayerParameter &operator[](size_t index) const {
    return *(node_list[index]);
  }

 private:
  void upgradev1net(const caffe::NetParameter &old,
                    caffe::NetParameter *dst) const {
    dst->CopyFrom(old);
    dst->clear_layers();
    dst->clear_layer();

    for (int i = 0; i < old.layers_size(); i++) {
      upgradev1layer(old.layers(i), dst->add_layer());
    }
  }

  const char *v1type2name(caffe::V1LayerParameter_LayerType type) const {
    switch (type) {
      case caffe::V1LayerParameter_LayerType_NONE: return "";
      case caffe::V1LayerParameter_LayerType_ABSVAL: return "AbsVal";
      case caffe::V1LayerParameter_LayerType_ACCURACY: return "Accuracy";
      case caffe::V1LayerParameter_LayerType_ARGMAX: return "ArgMax";
      case caffe::V1LayerParameter_LayerType_BNLL: return "BNLL";
      case caffe::V1LayerParameter_LayerType_CONCAT: return "Concat";
      case caffe::V1LayerParameter_LayerType_CONTRASTIVE_LOSS:
        return "ContrastiveLoss";
      case caffe::V1LayerParameter_LayerType_CONVOLUTION: return "Convolution";
      case caffe::V1LayerParameter_LayerType_DECONVOLUTION:
        return "Deconvolution";
      case caffe::V1LayerParameter_LayerType_DATA: return "Data";
      case caffe::V1LayerParameter_LayerType_DROPOUT: return "Dropout";
      case caffe::V1LayerParameter_LayerType_DUMMY_DATA: return "DummyData";
      case caffe::V1LayerParameter_LayerType_EUCLIDEAN_LOSS:
        return "EuclideanLoss";
      case caffe::V1LayerParameter_LayerType_ELTWISE: return "Eltwise";
      case caffe::V1LayerParameter_LayerType_EXP: return "Exp";
      case caffe::V1LayerParameter_LayerType_FLATTEN: return "Flatten";
      case caffe::V1LayerParameter_LayerType_HDF5_DATA: return "HDF5Data";
      case caffe::V1LayerParameter_LayerType_HDF5_OUTPUT: return "HDF5Output";
      case caffe::V1LayerParameter_LayerType_HINGE_LOSS: return "HingeLoss";
      case caffe::V1LayerParameter_LayerType_IM2COL: return "Im2col";
      case caffe::V1LayerParameter_LayerType_IMAGE_DATA: return "ImageData";
      case caffe::V1LayerParameter_LayerType_INFOGAIN_LOSS:
        return "InfogainLoss";
      case caffe::V1LayerParameter_LayerType_INNER_PRODUCT:
        return "InnerProduct";
      case caffe::V1LayerParameter_LayerType_LRN: return "LRN";
      case caffe::V1LayerParameter_LayerType_MEMORY_DATA: return "MemoryData";
      case caffe::V1LayerParameter_LayerType_MULTINOMIAL_LOGISTIC_LOSS:
        return "MultinomialLogisticLoss";
      case caffe::V1LayerParameter_LayerType_MVN: return "MVN";
      case caffe::V1LayerParameter_LayerType_POOLING: return "Pooling";
      case caffe::V1LayerParameter_LayerType_POWER: return "Power";
      case caffe::V1LayerParameter_LayerType_RELU: return "ReLU";
      case caffe::V1LayerParameter_LayerType_SIGMOID: return "Sigmoid";
      case caffe::V1LayerParameter_LayerType_SIGMOID_CROSS_ENTROPY_LOSS:
        return "SigmoidCrossEntropyLoss";
      case caffe::V1LayerParameter_LayerType_SILENCE: return "Silence";
      case caffe::V1LayerParameter_LayerType_SOFTMAX: return "Softmax";
      case caffe::V1LayerParameter_LayerType_SOFTMAX_LOSS:
        return "SoftmaxWithLoss";
      case caffe::V1LayerParameter_LayerType_SPLIT: return "Split";
      case caffe::V1LayerParameter_LayerType_SLICE: return "Slice";
      case caffe::V1LayerParameter_LayerType_TANH: return "TanH";
      case caffe::V1LayerParameter_LayerType_WINDOW_DATA: return "WindowData";
      case caffe::V1LayerParameter_LayerType_THRESHOLD: return "Threshold";
      default: throw nn_error("unknown v1 layer-type");
    }
  }

  void upgradev1layer(const caffe::V1LayerParameter &old,
                      caffe::LayerParameter *dst) const {
    dst->Clear();

    for (int i = 0; i < old.bottom_size(); i++) {
      dst->add_bottom(old.bottom(i));
    }

    for (int i = 0; i < old.top_size(); i++) {
      dst->add_top(old.top(i));
    }

    if (old.has_name()) dst->set_name(old.name());
    if (old.has_type()) dst->set_type(v1type2name(old.type()));

    for (int i = 0; i < old.blobs_size(); i++) {
      dst->add_blobs()->CopyFrom(old.blobs(i));
    }

    for (int i = 0; i < old.param_size(); i++) {
      while (dst->param_size() <= i) dst->add_param();
      dst->mutable_param(i)->set_name(old.param(i));
    }

#define COPY_PARAM(name)                                         \
  if (old.has_##name##_param()) {                                \
    dst->mutable_##name##_param()->CopyFrom(old.name##_param()); \
  }

    COPY_PARAM(accuracy);
    COPY_PARAM(argmax);
    COPY_PARAM(concat);
    COPY_PARAM(contrastive_loss);
    COPY_PARAM(convolution);
    COPY_PARAM(data);
    COPY_PARAM(dropout);
    COPY_PARAM(dummy_data);
    COPY_PARAM(eltwise);
    COPY_PARAM(exp);
    COPY_PARAM(hdf5_data);
    COPY_PARAM(hdf5_output);
    COPY_PARAM(hinge_loss);
    COPY_PARAM(image_data);
    COPY_PARAM(infogain_loss);
    COPY_PARAM(inner_product);
    COPY_PARAM(lrn);
    COPY_PARAM(memory_data);
    COPY_PARAM(mvn);
    COPY_PARAM(pooling);
    COPY_PARAM(power);
    COPY_PARAM(relu);
    COPY_PARAM(sigmoid);
    COPY_PARAM(softmax);
    COPY_PARAM(slice);
    COPY_PARAM(tanh);
    COPY_PARAM(threshold);
    COPY_PARAM(window_data);
    COPY_PARAM(transform);
    COPY_PARAM(loss);
#undef COPY_PARAM
  }

  caffe::NetParameter net;
  layer_node *root_node;
  /* layer name -> layer */
  std::map<std::string, layer_node *> layer_table;
  /* blob name -> bottom holder */
  std::map<std::string, layer_node *> blob_table;
  std::vector<layer_node> nodes;
  std::vector<const caffe::LayerParameter *> node_list;
};

}  // namespace detail
}  // namespace tiny_dnn

#ifdef _MSC_VER
#pragma warning(pop)
#endif
