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
      return -1;
    }
}

// parse caffe net and interpret as single layer vector
class tensorflow_node_vector {
 public:
    explicit tensorflow_node_vector(const caffe::NetParameter& net_orig)
            : net(net_orig) {
        if (net.layers_size() > 0) {
            upgradev1net(net_orig, &net);
        }

        nodes.reserve(net.layer_size());

        for (int i = 0; i < net.layer_size(); i++) {
            auto& l = net.layer(i);

            if (layer_table.find(l.name()) != layer_table.end()) continue;

            nodes.emplace_back(&l);
            layer_table[l.name()] = &nodes.back();
        }

        for (int i = 0; i < nodes.size(); i++) {
            auto& l = nodes[i];

            if (l.layer->bottom_size() > 0 && blob_table[l.layer->bottom(0)]) {
                auto& bottom = blob_table[l.layer->bottom(0)];
                l.prev = bottom;
                layer_table[bottom->layer->name()]->next = &l;
            }

            if (l.layer->top_size() > 0) {
                blob_table[l.layer->top(0)] = &l;
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

    const caffe::LayerParameter& operator[] (size_t index) const {
        return *(node_list[index]);
    }

 private:
    void upgradev1net(const caffe::NetParameter& old,
                      caffe::NetParameter *dst) const {
        dst->CopyFrom(old);
        dst->clear_layers();
        dst->clear_layer();

        for (int i = 0; i < old.layers_size(); i++) {
            upgradev1layer(old.layers(i), dst->add_layer());
        }
    }

    const char* v1type2name(caffe::V1LayerParameter_LayerType type) const {
        switch (type) {
        case caffe::V1LayerParameter_LayerType_NONE:
            return "";
        case caffe::V1LayerParameter_LayerType_ABSVAL:
            return "AbsVal";
        case caffe::V1LayerParameter_LayerType_ACCURACY:
            return "Accuracy";
        case caffe::V1LayerParameter_LayerType_ARGMAX:
            return "ArgMax";
        case caffe::V1LayerParameter_LayerType_BNLL:
            return "BNLL";
        case caffe::V1LayerParameter_LayerType_CONCAT:
            return "Concat";
        case caffe::V1LayerParameter_LayerType_CONTRASTIVE_LOSS:
            return "ContrastiveLoss";
        case caffe::V1LayerParameter_LayerType_CONVOLUTION:
            return "Convolution";
        case caffe::V1LayerParameter_LayerType_DECONVOLUTION:
            return "Deconvolution";
        case caffe::V1LayerParameter_LayerType_DATA:
            return "Data";
        case caffe::V1LayerParameter_LayerType_DROPOUT:
            return "Dropout";
        case caffe::V1LayerParameter_LayerType_DUMMY_DATA:
            return "DummyData";
        case caffe::V1LayerParameter_LayerType_EUCLIDEAN_LOSS:
            return "EuclideanLoss";
        case caffe::V1LayerParameter_LayerType_ELTWISE:
            return "Eltwise";
        case caffe::V1LayerParameter_LayerType_EXP:
            return "Exp";
        case caffe::V1LayerParameter_LayerType_FLATTEN:
            return "Flatten";
        case caffe::V1LayerParameter_LayerType_HDF5_DATA:
            return "HDF5Data";
        case caffe::V1LayerParameter_LayerType_HDF5_OUTPUT:
            return "HDF5Output";
        case caffe::V1LayerParameter_LayerType_HINGE_LOSS:
            return "HingeLoss";
        case caffe::V1LayerParameter_LayerType_IM2COL:
            return "Im2col";
        case caffe::V1LayerParameter_LayerType_IMAGE_DATA:
            return "ImageData";
        case caffe::V1LayerParameter_LayerType_INFOGAIN_LOSS:
            return "InfogainLoss";
        case caffe::V1LayerParameter_LayerType_INNER_PRODUCT:
            return "InnerProduct";
        case caffe::V1LayerParameter_LayerType_LRN:
            return "LRN";
        case caffe::V1LayerParameter_LayerType_MEMORY_DATA:
            return "MemoryData";
        case caffe::V1LayerParameter_LayerType_MULTINOMIAL_LOGISTIC_LOSS:
            return "MultinomialLogisticLoss";
        case caffe::V1LayerParameter_LayerType_MVN:
            return "MVN";
        case caffe::V1LayerParameter_LayerType_POOLING:
            return "Pooling";
        case caffe::V1LayerParameter_LayerType_POWER:
            return "Power";
        case caffe::V1LayerParameter_LayerType_RELU:
            return "ReLU";
        case caffe::V1LayerParameter_LayerType_SIGMOID:
            return "Sigmoid";
        case caffe::V1LayerParameter_LayerType_SIGMOID_CROSS_ENTROPY_LOSS:
            return "SigmoidCrossEntropyLoss";
        case caffe::V1LayerParameter_LayerType_SILENCE:
            return "Silence";
        case caffe::V1LayerParameter_LayerType_SOFTMAX:
            return "Softmax";
        case caffe::V1LayerParameter_LayerType_SOFTMAX_LOSS:
            return "SoftmaxWithLoss";
        case caffe::V1LayerParameter_LayerType_SPLIT:
            return "Split";
        case caffe::V1LayerParameter_LayerType_SLICE:
            return "Slice";
        case caffe::V1LayerParameter_LayerType_TANH:
            return "TanH";
        case caffe::V1LayerParameter_LayerType_WINDOW_DATA:
            return "WindowData";
        case caffe::V1LayerParameter_LayerType_THRESHOLD:
            return "Threshold";
        default:
            throw nn_error("unknown v1 layer-type");
        }
    }

    void upgradev1layer(const caffe::V1LayerParameter& old,
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

        #define COPY_PARAM(name) if (old.has_##name##_param()) dst->mutable_##name##_param()->CopyFrom(old.name##_param())

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
    std::map<std::string, layer_node*> layer_table;
    /* blob name -> bottom holder */
    std::map<std::string, layer_node*> blob_table;
    std::vector<layer_node> nodes;
    std::vector<const caffe::LayerParameter*> node_list;
};

} // namespace detail
} // namespace tiny_cnn
