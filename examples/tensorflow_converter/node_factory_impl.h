/*
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>

#include "graph.pb.h"

#include "tiny_cnn/layers/convolutional_layer.h"
#include "tiny_cnn/layers/deconvolutional_layer.h"
#include "tiny_cnn/layers/fully_connected_layer.h"
#include "tiny_cnn/layers/average_pooling_layer.h"
#include "tiny_cnn/layers/max_pooling_layer.h"
#include "tiny_cnn/layers/linear_layer.h"
#include "tiny_cnn/layers/lrn_layer.h"
#include "tiny_cnn/layers/dropout_layer.h"

using namespace std;
using namespace tensorflow;

typedef tiny_cnn::shape3d shape_t;

namespace tiny_cnn {
namespace detail {

template<class in_value>
void all2str(string & result, const in_value& t)
{
    ostringstream oss;
    oss<<t;
    result=oss.str();
}

/* This method is used as reference for bit operation for char* parsering

static int32_t BigEndInt(const char* b, int start) {
  return (((b[start + 1] & 0xff)) | ((b[ start+ 0] & 0xff)));
}
static float BigEndFloat(const char* b, int start) {

  float val=0;

  unsigned long result=0;
  result |= ((unsigned long)(b[start]) << 0x18);
  result |= ((unsigned long)(b[start+1]) << 0x10);
  result |= ((unsigned long)(b[start+2]) << 0x08);
  result |= ((unsigned long)(b[start+3]));
  memcpy(&val,&result,4);

  return val;
}
*/

static float char_to_float(const char* b) {
  float f;
  memcpy(&f, b, sizeof(float));
  return f;
}

static float char_to_int32(const char* b) {
  int32_t i;
  memcpy(&i, b, sizeof(int32_t));
  return i;
}

void summarize_attr_value(const string& attr_string, const AttrValue& attr_value) {
  switch (attr_value.value_case()) {
    case AttrValue::kS: {
      string tmp_tostr;
      all2str(tmp_tostr, attr_value.s());
      cout << "   (kString) " << attr_string << ": " << tmp_tostr << endl;
    }
      break;
    case AttrValue::kI: {
      string tmp_tostr;
      all2str(tmp_tostr, attr_value.i());
      cout << "   (kInt) " << attr_string << ": " << tmp_tostr << endl;
    }
      break;
    case AttrValue::kF: {
      string tmp_tostr;
      all2str(tmp_tostr, attr_value.f());
      cout << "   (kFloat) " << attr_string << ": " << tmp_tostr << endl;
    }
      break;
    case AttrValue::kB: {
      string tmp_tostr;
      all2str(tmp_tostr, attr_value.b());
      cout << "   (kBool) " << attr_string << ": " << tmp_tostr << endl;
    }
      break;
    case AttrValue::kType: {
      string tmp_tostr;
      all2str(tmp_tostr, attr_value.type());
      cout << "   (kType) " << attr_string << ": " << tmp_tostr << endl;
    }
      break;
    case AttrValue::kShape: {
      string tmp_tostr;
      cout << "   (kShape) " << attr_string << ": ";
      if (attr_value.shape().unknown_rank()) {
        cout << "<unknown>";
      }
      string s = "[";
      bool first = true;
      for (const auto& d : attr_value.shape().dim()) {
        if (d.size() == -1) {
          s += (first ? "" : ",");
          s += "?";
        }
        else {
          s += (first ? "" : ",");
          all2str(tmp_tostr, d.size());
          s += tmp_tostr;
        }
        first = false;
      }
      s += "]";
      cout << s << endl;
    }
      break;
    case AttrValue::kTensor: {
      string tmp_tostr;
      /* Both way is a TensorFlow representatio of all Tensor information
      cout << "   (kTensor) " << attr_string << ": " << attr_value.tensor().ShortDebugString();
      cout << "   (kTensor) " << attr_string << ": " << attr_value.tensor().DebugString();
      */
      if (attr_value.tensor().dtype() == DT_FLOAT) {
        cout << "   (kTensor) " << attr_string << ": [";
        for (int i = 0; i < attr_value.tensor().tensor_content().size()/4; i++)
          cout << char_to_float(attr_value.tensor().tensor_content().c_str() + 4*i) << ' ';
        cout << ']' << endl;
      } else if (attr_value.tensor().dtype() == DT_INT32){
        cout << "   (kTensor) " << attr_string << ": [";
        for (int i = 0; i < attr_value.tensor().tensor_content().size()/4; i++)
          cout << char_to_int32(attr_value.tensor().tensor_content().c_str() + 4*i) << ' ';
        cout << ']' << endl;

      }
      cout << "   kTensor Size: " << attr_string << ": ";
      for (const auto& d : attr_value.tensor().tensor_shape().dim()) {
        if (d.size() == -1)
          cout << "What? Size is undefined here";
        else {
          all2str(tmp_tostr, d.size());
          cout << ' ' << tmp_tostr;
        }
      }
      cout << endl;
      break;
    }
    case AttrValue::kList: {
      string tmp_tostr;
      cout << "   (kList) " << attr_string << ": ";
      string ret = "[";
      if (attr_value.list().s_size() > 0) {
        for (int i = 0; i < attr_value.list().s_size(); ++i) {
          if (i > 0) ret += ", ";
          ret += attr_value.list().s(i);
        }
      } else if (attr_value.list().i_size() > 0) {
        for (int i = 0; i < attr_value.list().i_size(); ++i) {
          if (i > 0) ret += ", ";
          all2str(tmp_tostr, attr_value.list().i(i));
          ret += tmp_tostr;
        }
      } else if (attr_value.list().f_size() > 0) {
        for (int i = 0; i < attr_value.list().f_size(); ++i) {
          if (i > 0) ret += ", ";
          ret += attr_value.list().f(i);
        }
      } else if (attr_value.list().b_size() > 0) {
        for (int i = 0; i < attr_value.list().b_size(); ++i) {
          if (i > 0) ret += ", ";
          ret += (attr_value.list().b(i) ? "true" : "false");
        }
      } else if (attr_value.list().type_size() > 0) {
        for (int i = 0; i < attr_value.list().type_size(); ++i) {
          if (i > 0) ret += ", ";
          ret += attr_value.list().type(i);
        }
      } else if (attr_value.list().shape_size() > 0) {
        for (int i = 0; i < attr_value.list().shape_size(); ++i) {
          if (i > 0) ret += ", ";
          for (const auto& d : attr_value.list().shape(i).dim()){
            if (d.size() == -1)
              ret += "   <What? Size is undefined here>";
            else {
              ret += ' ';
              ret += d.size();
            }
          }
        }
      } else if (attr_value.list().tensor_size() > 0) {
        for (int i = 0; i < attr_value.list().tensor_size(); ++i) {
          if (i > 0) ret += ", ";
          ret += attr_value.list().tensor(i).ShortDebugString();
        }
      }

      ret += "]";
      cout << ret << endl;
      break;
    }
    case AttrValue::kFunc: {
      string tmp_tostr;
      for (auto p : attr_value.func().attr()) {
        cout << "   SubAttr: " << p.first << ": " << endl;
        summarize_attr_value(p.first, p.second);
      }
      break;
    }
    case AttrValue::kPlaceholder: {
      string tmp_tostr;
      all2str(tmp_tostr, attr_value.placeholder());
      cout << "   (kPlaceholder) " << attr_string << ": " << tmp_tostr << endl;
    }
      break;
    case AttrValue::VALUE_NOT_SET: {
      cout << "   <Unknown AttrValue type> " << endl;
    }
      break;
  }
}

bool IsPlaceholder(const tensorflow::NodeDef& node_def) {
  if (node_def.op() != "Placeholder" || node_def.name() != "feed") {
    return false;
  }
  bool found_dtype = false;
  bool found_shape = false;
  for (const auto& attr : node_def.attr()) {
    if (attr.first == "dtype") {
      if (attr.second.type() == tensorflow::DT_INT32) {
        found_dtype = true;
      } else {
        return false;
      }
    } else if (attr.first == "shape") {
      found_shape = true;
    }
  }
  return found_dtype && found_shape;
}

bool IsScalarConst(const tensorflow::NodeDef& node_def) {
  if (node_def.op() != "Const" || node_def.name() != "scalar") {
    return false;
  }
  bool found_dtype = false;
  bool found_value = false;
  for (const auto& attr : node_def.attr()) {
    if (attr.first == "dtype") {
      if (attr.second.type() == tensorflow::DT_INT32) {
        found_dtype = true;
      } else {
        return false;
      }
    } else if (attr.first == "value") {
      if (attr.second.has_tensor() &&
          attr.second.tensor().int_val_size() > 0) {
        found_value = true;
      } else {
        return false;
      }
    }
  }
  return found_dtype && found_value;
}

// Iterates though all nodes in the GraphDef and prints info about them.
void ListNodes(const tensorflow::GraphDef& graph_def) {
  cout << "  Node Size: " << graph_def.node_size() << endl;
  for (int i = 0; i < graph_def.node_size(); i++) {
    const tensorflow::NodeDef& node_def = graph_def.node(i);
    // print nodes information
    fprintf(stderr, "\nNode: %s (%s)\n", node_def.name().c_str(), node_def.op().c_str());
    fprintf(stderr, "  Is Placeholder: %d, Is Const: %d\n", IsPlaceholder(node_def), IsScalarConst(node_def));

    cout << "  Inputs: ";
    for (const string& input : node_def.input()) {
      cout << input << ' ';
    }
    cout << endl;

    cout << " Attrbutes are shown below:" << endl;
    for (const auto& attr : node_def.attr()) {
      auto iter = node_def.attr().find(attr.first);
      summarize_attr_value(attr.first, iter->second);
    }
  }
}

inline void read_proto_from_binary(const std::string& protobinary,
                                   tensorflow::GraphDef *graph_def) {


    // Read the existing graph.
    fstream input(protobinary, ios::in | ios::binary);
    if (!graph_def->ParseFromIstream(&input)) {
      cerr << "Failed to parse graph." << endl;
    }
}

inline std::shared_ptr<layer> create_max_pool(int pool_size,
                                              int stride,
                                              const shape_t& bottom_shape,
                                              shape_t *top_shape) {
    using max_pool = max_pooling_layer<activation::identity>;
    auto mp = std::make_shared<max_pool>(bottom_shape.width_,
                                         bottom_shape.height_,
                                         bottom_shape.depth_,
                                         pool_size, stride);
    // TODO
    //  *top_shape = mp->out_shape();
    *top_shape = mp->out_shape()[0];  // check this
    mp->init_weight();

    return mp;
}

inline std::shared_ptr<layer> create_ave_pool(int pool_size,
                                              int stride,
                                              const shape_t& bottom_shape,
                                              shape_t *top_shape) {
    using ave_pool = average_pooling_layer<activation::identity>;
    auto ap = std::make_shared<ave_pool>(bottom_shape.width_,
                                         bottom_shape.height_,
                                         bottom_shape.depth_,
                                         pool_size, stride);

    // tiny-cnn has trainable parameter in average-pooling layer
    float_t weight = float_t(1) / sqr(pool_size);

    vec_t& w = *ap->get_weights()[0];
    vec_t& b = *ap->get_weights()[1];

    //std::fill(ap->weight().begin(), ap->weight().end(), weight);
    //std::fill(ap->bias().begin(), ap->bias().end(), float_t(0));

    std::fill(w.begin(), w.end(), weight);
    std::fill(b.begin(), b.end(), float_t(0));

    // TODO: check if this works
    *top_shape = ap->out_shape()[0];
    ap->init_weight();

    return ap;
}

inline
std::shared_ptr<layer> create_softmax(const tensorflow::NodeDef& layer,
                                      const shape_t& bottom_shape, shape_t *) {
    auto sm = std::make_shared<linear_layer<activation::softmax>>(
        bottom_shape.size());
    return sm;
}

inline
std::shared_ptr<layer> create_sigmoid(const tensorflow::NodeDef& layer,
                                      const shape_t& bottom_shape, shape_t *) {
    auto ce = std::make_shared<linear_layer<activation::sigmoid>>(
        bottom_shape.size());
    return ce;
}

inline
std::shared_ptr<layer> create_tanh(const tensorflow::NodeDef& layer,
                                   const shape_t& bottom_shape, shape_t *) {
    auto tanh = std::make_shared<linear_layer<activation::tan_h>>(
        bottom_shape.size());
    return tanh;
}

inline
std::shared_ptr<layer> create_pooling(const tensorflow::NodeDef& layer,
                                      const shape_t& bottom_shape,
                                      shape_t *top_shape) {
    if (!layer.has_pooling_param()) {
        throw nn_error("pool param missing");
    }

    auto pool_param = layer.pooling_param();

    layer_size_t h_stride = 0;
    layer_size_t w_stride = 0;
    layer_size_t pool_size = 0;

    if (!get_kernel_size_2d(pool_param, &pool_size)) {
        pool_size = pool_param.kernel_size();
    }

    if (pool_param.has_stride() || pool_param.has_stride_h()) {
        h_stride = pool_param.has_stride() ?
                   pool_param.stride() : pool_param.stride_h();
    }

    if (pool_param.has_stride() || pool_param.has_stride_w()) {
        w_stride = pool_param.has_stride() ?
                   pool_param.stride() : pool_param.stride_w();
    }

    if (h_stride != w_stride) {  // || h_stride != pool_size)
        throw nn_error("unsupported pool shape");
    }

    if (pool_param.has_pool()) {
        auto type = pool_param.pool();

        switch (type) {
            case caffe::PoolingParameter_PoolMethod_MAX:
                return create_max_pool(pool_size, h_stride,
                                       bottom_shape, top_shape);
            case caffe::PoolingParameter_PoolMethod_AVE:
                return create_ave_pool(pool_size, h_stride,
                                       bottom_shape, top_shape);
            default:
                throw nn_error("unsupported layer type");
        }
    }

    // default: max-pool
    return create_max_pool(pool_size, h_stride, bottom_shape, top_shape);
}

inline
std::shared_ptr<layer> create_relu(const tensorflow::NodeDef& layer,
                                   const shape_t& bottom_shape, shape_t *) {
    auto relu = std::make_shared<linear_layer<activation::relu>>(
        bottom_shape.size());
    return relu;
}

inline std::shared_ptr<layer> create_batchnorm(const tensorflow::NodeDef& layer,
    const shape_t& bottom_shape, shape_t *top_shape) {
    using bn_layer = batch_normalization_layer;

    *top_shape = bottom_shape;

    float_t eps = 1e-5;
    float_t momentum = 0.999;

    if (layer.has_batch_norm_param()) {
        auto bn_param = layer.batch_norm_param();

        if (bn_param.has_eps()) {
            eps = bn_param.eps();
        }
        if (bn_param.has_moving_average_fraction()) {
            momentum = bn_param.moving_average_fraction();
        }
    }

    auto bn = std::make_shared<bn_layer>(bottom_shape.area(), bottom_shape.depth_, eps, momentum, net_phase::test);

    // weight
    if (layer.blobs_size() > 0) {
        auto global_stats = layer.blobs();
        if (global_stats.size() != 3) {
            throw std::runtime_error("unexpected bn stored statistics");
        }

        float_t scale_factor = global_stats.Get(2).data(0) == 0 ? 0 : 1 / global_stats.Get(2).data(0);
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


inline void load_weights_fullyconnected(const tensorflow::NodeDef& src,
                                        layer *dst) {
    auto weights = src.blobs(0);
    int curr = 0;

    if (dst->out_size() * dst->in_size() !=
        static_cast<cnn_size_t>(weights.data_size())) {
        throw nn_error(
            std::string("layer size mismatch!") +
            "caffe(" + src.name() + "):" + to_string(weights.data_size()) + "\n" +
            "tiny-cnn(" + dst->layer_type() + "):" + to_string(dst->get_weights().size()));
    }

    vec_t& w = *dst->get_weights()[0];
    vec_t& b = *dst->get_weights()[1];

    // fill weights
    for (size_t o = 0; o < dst->out_size(); o++) {
        for (size_t i = 0; i < dst->in_size(); i++) {
            // TODO: how to access to weights?
            //dst->weight()[i * dst->out_size() + o] = weights.data(curr++); // transpose
            w[i * dst->out_size() + o] = weights.data(curr++); // transpose
        }
    }

    // fill bias
    if (src.inner_product_param().bias_term()) {
        auto biases = src.blobs(1);
        for (size_t o = 0; o < dst->out_size(); o++) {
            // TODO: how to access to biases?
            //dst->bias()[o] = biases.data(o);
            b[o] = biases.data(o);
        }
    }
}

inline std::shared_ptr<layer> create_fullyconnected(
        const tensorflow::NodeDef& layer,
        const shape_t& bottom_shape, shape_t *top_shape) {
    using fc_layer = fully_connected_layer<activation::identity>;

    if (!layer.has_inner_product_param()) {
        throw nn_error("inner-product param missing");
    }

    layer_size_t dim_input = 0, dim_output = 0;
    bool has_bias = true;

    auto ip_param = layer.inner_product_param();
    has_bias = ip_param.bias_term();

    dim_output = ip_param.num_output();
    dim_input = bottom_shape.size();

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

    // TODO: check if it works
    *top_shape = ip->out_shape()[0];
    return ip;
}

inline void load_weights_conv(const tensorflow::NodeDef& src, layer *dst) {
    // fill weight
    auto weights = src.blobs(0);

    //TODO: check if it works
    //int out_channels = dst->out_shape().depth_;
    //int in_channels = dst->in_shape().depth_;
    int out_channels = dst->out_data_shape()[0].depth_;
    int in_channels = dst->in_data_shape()[0].depth_;

    connection_table table;
    auto conv_param = src.convolution_param();
    int dst_idx = 0;
    int src_idx = 0;
    int window_size = get_kernel_size_2d(conv_param);

    if (conv_param.has_group()) {
        table = connection_table(conv_param.group(), in_channels, out_channels);
    }

    vec_t& w = *dst->get_weights()[0];
    vec_t& b = *dst->get_weights()[1];

    // fill weights
    for (int o = 0; o < out_channels; o++) {
        for (int i = 0; i < in_channels; i++) {
            if (!table.is_connected(o, i)) {
                dst_idx += window_size * window_size;
                continue;
            }
            for (int x = 0; x < window_size * window_size; x++) {
                //TODO
                //dst->weight()[dst_idx++] = weights.data(src_idx++);
                w[dst_idx++] =  weights.data(src_idx++);
            }
        }
    }

    // fill bias
    if (conv_param.bias_term()) {
        auto biases = src.blobs(1);
        for (int o = 0; o < out_channels; o++) {
            //TODO
            //dst->bias()[o] = biases.data(o);
            b[o] = biases.data(o);
        }
    }
}

inline void load_weights_deconv(const tensorflow::NodeDef& src, layer *dst) {
    // fill weight
    auto weights = src.blobs(0);

    //TODO: check if it works
    //int out_channels = dst->out_shape().depth_;
    //int in_channels = dst->in_shape().depth_;
    int out_channels = dst->out_data_shape()[0].depth_;
    int in_channels = dst->in_data_shape()[0].depth_;

    connection_table table;
    auto deconv_param = src.convolution_param();
    int dst_idx = 0;
    int src_idx = 0;
    int window_size = get_kernel_size_2d(deconv_param);

    if (deconv_param.has_group()) {
        table = connection_table(deconv_param.group(), in_channels, out_channels);
    }

    vec_t& w = *dst->get_weights()[0];
    vec_t& b = *dst->get_weights()[1];

    // fill weights
    for (int o = 0; o < out_channels; o++) {
        for (int i = 0; i < in_channels; i++) {
            if (!table.is_connected(o, i)) {
                dst_idx += window_size * window_size;
                continue;
            }
            for (int x = 0; x < window_size * window_size; x++) {
                //TODO
                //dst->weight()[dst_idx++] = weights.data(src_idx++);
                w[dst_idx++] =  weights.data(src_idx++);
            }
        }
    }

    // fill bias
    if (deconv_param.bias_term()) {
        auto biases = src.blobs(1);
        for (int o = 0; o < out_channels; o++) {
            //TODO
            //dst->bias()[o] = biases.data(o);
            b[o] = biases.data(o);
        }
    }
}

inline void load_weights_pool(const tensorflow::NodeDef& src, layer *dst) {
    auto pool_param = src.pooling_param();

    //TODO
    //if (dst->weight().size()) {
    if (dst->get_weights().size()) {
        layer_size_t pool_size = 0;

        if (!get_kernel_size_2d(pool_param, &pool_size)) {
            pool_size = pool_param.kernel_size();
        }

        // tiny-cnn has trainable parameter in average-pooling layer
        float_t weight = float_t(1) / sqr(pool_size);

        //TODO
        /*if (!dst->weight().empty()) {
            std::fill(dst->weight().begin(), dst->weight().end(), weight);
        }
        if (!dst->bias().empty()) {
            std::fill(dst->bias().begin(), dst->bias().end(), float_t(0));
            dst->init_bias();
        }*/

        vec_t& w = *dst->get_weights()[0];
        vec_t& b = *dst->get_weights()[1];

        if (!w.empty()) {
            std::fill(w.begin(), w.end(), weight);
        }
        if (!b.empty()) {
            std::fill(b.begin(), b.end(), float_t(0));
            //dst->init_bias();
        }
    }
}

inline
std::shared_ptr<layer> create_lrn(const tensorflow::NodeDef& layer,
                                  const shape_t& bottom_shape,
                                  shape_t *top_shape) {
    using lrn_layer = lrn_layer<activation::identity>;

    if (!layer.has_lrn_param()) {
        throw nn_error("lrn param missing");
    }

    auto lrn_param = layer.lrn_param();
    layer_size_t local_size = 5;
    float_t alpha = 1;
    float_t beta = 5;
    norm_region region = norm_region::across_channels;

    if (lrn_param.has_local_size()) local_size = lrn_param.local_size();
    if (lrn_param.has_alpha()) alpha = lrn_param.alpha();
    if (lrn_param.has_beta()) beta = lrn_param.beta();
    if (lrn_param.has_norm_region()) {
        if (lrn_param.norm_region() == caffe::LRNParameter_NormRegion_WITHIN_CHANNEL) // NOLINT
            region = norm_region::within_channels;
    }

    auto lrn = std::make_shared<lrn_layer>(bottom_shape.width_,
                                           bottom_shape.height_,
                                           local_size,
                                           bottom_shape.depth_,
                                           alpha, beta, region);
    return lrn;
}

inline
std::shared_ptr<layer> create_dropout(const tensorflow::NodeDef& layer,
                                      const shape_t& bottom_shape,
                                      shape_t *top_shape) {
    if (!layer.has_dropout_param()) {
        throw nn_error("dropout param missing");
    }

    float_t dropout_rate = float_t(0.5);

    if (layer.dropout_param().has_dropout_ratio()) {
        dropout_rate = layer.dropout_param().dropout_ratio();
    }

    auto dropout = std::make_shared<dropout_layer>(bottom_shape.size(),
                                                   dropout_rate,
                                                   net_phase::test);
    return dropout;
}

inline
std::shared_ptr<layer> create_convlayer(const tensorflow::NodeDef& layer,
                                        const shape_t& bottom_shape,
                                        shape_t *top_shape) {
    using conv_layer = convolutional_layer<activation::identity>;

    if (!layer.has_convolution_param()) {
        throw nn_error("convolution param missing");
    }

    // layer parameters
    layer_size_t in_width = 0, in_height = 0, window_size = 0;
    layer_size_t in_channels = 0, out_channels = 0;
    layer_size_t w_stride = 1, h_stride = 1;
    bool has_bias = true;
    padding pad_type = padding::valid;
    connection_table table;

    auto conv_param = layer.convolution_param();

    // shape
    out_channels = conv_param.num_output();
    in_channels = bottom_shape.depth_;
    in_width = bottom_shape.width_;
    in_height = bottom_shape.height_;
    has_bias = conv_param.bias_term();
    window_size = get_kernel_size_2d(conv_param);

    // padding
    if (conv_param.pad_size() == 1 ||
       (conv_param.has_pad_w() && conv_param.has_pad_h())) {
        uint32_t pad_w = conv_param.pad_size() == 1 ?
                         conv_param.pad(0) : conv_param.pad_w();

        uint32_t pad_h = conv_param.pad_size() == 1 ?
                         conv_param.pad(0) : conv_param.pad_h();

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
        h_stride = conv_param.stride_size() == 1 ?
                   conv_param.stride(0) : conv_param.stride_h();
    }

    if (conv_param.stride_size() == 1 || conv_param.has_stride_w()) {
        w_stride = conv_param.stride_size() == 1 ?
                   conv_param.stride(0) : conv_param.stride_w();
    }

    // group
    if (conv_param.has_group()) {
        table = connection_table(conv_param.group(), in_channels, out_channels);
    }

    auto conv = std::make_shared<conv_layer>(in_width, in_height,
                                             window_size,
                                             in_channels, out_channels,
                                             table,
                                             pad_type,
                                             has_bias,
                                             w_stride, h_stride);
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
    //TODO
    //*top_shape = conv->out_shape();
    *top_shape = conv->out_shape()[0];
    return conv;
}

inline
std::shared_ptr<layer> create_deconvlayer(const tensorflow::NodeDef& layer,
                                        const shape_t& bottom_shape,
                                        shape_t *top_shape) {
    using deconv_layer = deconvolutional_layer<activation::identity>;

    if (!layer.has_convolution_param()) {
        throw nn_error("deconvolution param missing");
    }

    // layer parameters
    layer_size_t in_width = 0, in_height = 0, window_size = 0;
    layer_size_t in_channels = 0, out_channels = 0;
    layer_size_t w_stride = 1, h_stride = 1;
    bool has_bias = true;
    padding pad_type = padding::valid;
    connection_table table;

    auto deconv_param = layer.convolution_param();

    // shape
    out_channels = deconv_param.num_output();
    in_channels = bottom_shape.depth_;
    in_width = bottom_shape.width_;
    in_height = bottom_shape.height_;
    has_bias = deconv_param.bias_term();
    window_size = get_kernel_size_2d(deconv_param);

    // unpadding
    if (deconv_param.pad_size() == 1 ||
       (deconv_param.has_pad_w() && deconv_param.has_pad_h())) {
        uint32_t unpad_w = deconv_param.pad_size() == 1 ?
                         deconv_param.pad(0) : deconv_param.pad_w();

        uint32_t unpad_h = deconv_param.pad_size() == 1 ?
                         deconv_param.pad(0) : deconv_param.pad_h();

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
        h_stride = deconv_param.stride_size() == 1 ?
                   deconv_param.stride(0) : deconv_param.stride_h();
    }

    if (deconv_param.stride_size() == 1 || deconv_param.has_stride_w()) {
        w_stride = deconv_param.stride_size() == 1 ?
                   deconv_param.stride(0) : deconv_param.stride_w();
    }

    // group
    if (deconv_param.has_group()) {
        table = connection_table(deconv_param.group(), in_channels, out_channels);
    }

    auto deconv = std::make_shared<deconv_layer>(in_width, in_height,
                                             window_size,
                                             in_channels, out_channels,
                                             table,
                                             pad_type,
                                             has_bias,
                                             w_stride, h_stride);
    // filler
    if (deconv_param.has_weight_filler()) {
        deconv->weight_init(create_filler(deconv_param.weight_filler().type()));
    }

    if (deconv_param.has_bias_filler()) {
        deconv->bias_init(create_filler(deconv_param.bias_filler().type()));
    }

    // set weight (optional)
    if (layer.blobs_size() > 0) {  // blobs(0)...weight, blobs(1)...bias
        load_weights_deconv(layer, deconv.get());
    }
    //TODO
    //*top_shape = deconv->out_shape();
    *top_shape = deconv->out_shape()[0];
    return deconv;
}

inline bool layer_skipped(const std::string& type) {
    if (type == "Data" || type == "EuclideanLoss" || type == "Input") return true;
    return false;
}

inline bool layer_has_weights(const std::string& type) {
    static const char* activations[] = {
        "SoftmaxWithLoss", "SigmoidCrossEntropyLoss", "LRN", "Dropout",
        "ReLU", "Sigmoid", "TanH", "Softmax"
    };
    for (unsigned int i = 0; i < sizeof(activations) / sizeof(activations[0]); i++) {
        if (activations[i] == type) return false;
    }
    return true;
}

inline bool layer_supported(const std::string& type) {
    static const char* supported[] = {
        "InnerProduct", "Convolution", "Deconvolution", "Pooling",
        "LRN", "Dropout",
        "SoftmaxWithLoss", "SigmoidCrossEntropyLoss",
        "ReLU", "Sigmoid", "TanH", "Softmax", "BatchNorm"
    };

    for (size_t i = 0; i < sizeof(supported) / sizeof(supported[0]); i++) {
        if (supported[i] == type) return true;
    }
    return false;
}

inline bool layer_match(const std::string& caffetype,
                        const std::string& tiny_cnn_type) {
    const char* conversions[][2] = {
        { "InnerProduct", "fully-connected" },
        { "Convolution", "conv" },
        { "Deconvolution", "deconv" },
        { "Pooling", "ave-pool" },
        { "Pooling", "max-pool" }
    };

    for (size_t i = 0; i < sizeof(conversions) / sizeof(conversions[0]); i++) {
        if (conversions[i][0] == caffetype &&
            conversions[i][1] == tiny_cnn_type) return true;
    }
    return false;
}

inline std::shared_ptr<layer> create(const NodeDef& node,
                                     const shape_t& in_shape,
                                     shape_t* out_shape) {
    const std::string layer_type = node.op();

    if (layer_type == "Conv2D") {
        return detail::create_convlayer(node, in_shape, out_shape);
    }

    if (layer_type == "Conv2D") {
        return detail::create_deconvlayer(node, in_shape, out_shape);
    }

    if (layer_type == "InnerProduct") {
        return detail::create_fullyconnected(node, in_shape, out_shape);
    }

    if (layer_type == "Pooling") {
        return detail::create_pooling(node, in_shape, out_shape);
    }

    if (layer_type == "BatchNorm") {
        return detail::create_batchnorm(node, in_shape, out_shape);
    }

    if (layer_type == "LRN") {
        return detail::create_lrn(node, in_shape, out_shape);
    }

    if (layer_type == "Dropout") {
        return detail::create_dropout(node, in_shape, out_shape);
    }

    if (layer_type == "SoftmaxWithLoss" ||
        layer_type == "Softmax") {
        return detail::create_softmax(node, in_shape, out_shape);
    }

    if (layer_type == "SigmoidCrossEntropyLoss" ||
        layer_type == "Sigmoid") {
        return detail::create_sigmoid(node, in_shape, out_shape);
    }

    if (layer_type == "ReLU") {
        return detail::create_relu(node, in_shape, out_shape);
    }

    if (layer_type == "TanH") {
        return detail::create_tanh(node, in_shape, out_shape);
    }

    throw nn_error("node parser not found");

}

inline void load(const tensorflow::NodeDef& src, layer *dst) {
    typedef std::function<void(const tensorflow::NodeDef&, layer*)> factoryimpl; // NOLINT
    std::unordered_map<std::string, factoryimpl> factory_registry;

    factory_registry["Convolution"] = detail::load_weights_conv;
    factory_registry["Deconvolution"] = detail::load_weights_deconv;
    factory_registry["InnerProduct"] = detail::load_weights_fullyconnected;
    factory_registry["Pooling"] = detail::load_weights_pool;

    if (factory_registry.find(src.type()) == factory_registry.end()) {
        throw nn_error("layer parser not found");
    }

    return factory_registry[src.type()](src, dst);
}

struct layer_node {
    const tensorflow::NodeDef *node;
    const vector<layer_node*> *prev;  // bottom-side

    layer_node() : node(0), prev(0) {}
    explicit layer_node(const tensorflow::NodeDef *l)
        : node(l), prev(0) {}
};

// parse caffe net and interpret as single layer vector
class tensorflow_node_vector {
 public:
    explicit tensorflow_node_vector(const tensorflow::GraphDef& net_orig)
            : graph(net_orig) {

        nodes.reserve(graph.node_size());

        for (int i = 0; i < graph.node_size(); i++) {
            auto& n = graph_def.node(i);

            if (layer_table.find(n.name()) != layer_table.end()) continue;

            nodes.emplace_back(&n);
            layer_table[n.name()] = &nodes.back();
        }

        for (int i = 0; i < nodes.size(); i++) {
            auto& n = nodes[i];

            if (n.node.input().size() > 0) {
                for (const string& input : n.node.input()) {
                    if (node_table[input]) {
                        auto& bottom = node_table[input];
                        n.prev.push_back(bottom);
                        // layer_table[bottom->layer->name()]->next = &n;
                    }
                    /*
                    if (n.layer->top_size() > 0) {
                        node_table[n.layer->top(0)] = &n;
                    }
                    */
                }
            }
        }

        auto root = std::find_if(nodes.begin(),
                                 nodes.end(), [](const layer_node& n) {
            return n.prev == 0;
        });

        if (root == nodes.end()) {
            throw nn_error("root layer not found");
        }

        root_node = &*root;
        const layer_node *current = &*root;

        while (current) {
            node_list.push_back(current->layer);
            current = current->next;
        }
    }

    size_t size() const {
        return node_list.size();
    }

    const tensorflow::NodeDef& operator[] (size_t index) const {
        return *(node_list[index]);
    }

 private:

    tensorflow::GraphDef graph;
    layer_node *root_node;
    /* layer name -> layer */
    std::map<std::string, layer_node*> layer_table;
    /* blob name -> bottom holder */
    std::map<std::string, vector<layer_node*>*> node_table;
    std::vector<layer_node> nodes;
    std::vector<const tensorflow::NodeDef*> node_list;
};

} // namespace detail
} // namespace tiny_cnn
