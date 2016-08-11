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
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include "graph.pb.h"
#include "tiny_cnn/network.h"
#include "tiny_cnn/lossfunctions/loss_function.h"
#include "tiny_cnn/optimizers/optimizer.h"
#include "tiny_cnn/util/util.h"
#include "examples/tensorflow_convertor/node_factory_impl.h"

using namespace std;
using namespace tensorflow;

namespace tiny_cnn {

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


/**
* create whole network and load weights from caffe's netparameter
*
* @param layer [in] netparameter of caffemodel
* @param data_shape [in] size of input data (width x height x channels)
*/
inline std::shared_ptr<network<sequential>>
create_net_from_caffe_net(const tensorflow::GraphDef& layer, const shape3d& data_shape)
{
    detail::tensorflow_node_vector src_net(layer);
    shape_t shape;

    if (data_shape.size() > 0) {
        shape = data_shape;
    } else {
        if (layer.input_shape_size() > 0) {
            // input_shape is deprecated in Caffe
            // blob dimensions are ordered by number N x channel K x height H x width W
            int depth  = static_cast<int>(layer.input_shape(0).dim(1));
            int height = static_cast<int>(layer.input_shape(0).dim(2));
            int width  = static_cast<int>(layer.input_shape(0).dim(3));
            shape = shape3d(width, height, depth);
        }
        else if (src_net[0].has_input_param()) {
            // blob dimensions are ordered by number N x channel K x height H x width W
            int depth  = static_cast<int>(src_net[0].input_param().shape(0).dim(1));
            int height = static_cast<int>(src_net[0].input_param().shape(0).dim(2));
            int width  = static_cast<int>(src_net[0].input_param().shape(0).dim(3));
            shape = shape3d(width, height, depth);
        }
        else {
            throw nn_error("input_shape not found in caffemodel. must specify input shape explicitly");
        }
    }

    auto dst_net = std::make_shared<network<sequential>>(layer.name());

    for (size_t i = 0; i < src_net.size(); i++) {
        auto type = src_net[i].type();

        if (detail::layer_skipped(type)) {
            continue;
        }

        if (!detail::layer_supported(type)) {
            throw nn_error("error: tiny-cnn does not support this layer type:" + type);
        }

        shape_t shape_next = shape;
        auto layer = detail::create(src_net[i], shape, &shape_next);

        nn_info("convert " + type + " => " + typeid(*layer).name());
        nn_info("shape:" + to_string(shape_next));

        *dst_net << layer;
        shape = shape_next;
    }

    return dst_net;
}


/**
 * create whole network and load weights from tensorflow's netparameter
 *
 * @param layer [in] netparameter of caffemodel
 * @param data_shape [in] size of input data (width x height x channels)
 */
inline std::shared_ptr<network<sequential>>
create_net_from_caffe_protobinary(const std::string& tensorflowbinarymodel, const shape3d& data_shape)
{
    tensorflow::GraphDef np;

    detail::read_proto_from_binary(tensorflowbinarymodel, &np);
    return create_net_from_caffe_net(np, data_shape);
}

}