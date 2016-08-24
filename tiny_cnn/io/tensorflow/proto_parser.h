#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include "tensorflow/core/framework/graph.pb.h"

namespace tiny_cnn {

template<class in_value>
inline void all2str(std::string & result, const in_value& t)
{
    std::ostringstream oss;
    oss<<t;
    result=oss.str();
}

inline float char_to_float(const char* b) {
  float f;
  memcpy(&f, b, sizeof(float));
  return f;
}

inline float char_to_int32(const char* b) {
  int32_t i;
  memcpy(&i, b, sizeof(int32_t));
  return i;
}

inline void summarize_attr_value(const std::string& attr_string, const tensorflow::AttrValue& attr_value) {
  switch (attr_value.value_case()) {
    case tensorflow::AttrValue::kS: {
      std::string tmp_tostr;
      all2str(tmp_tostr, attr_value.s());
      std::cout << "   (kString) " << attr_string << ": " << tmp_tostr << std::endl;
    }
      break;
    case tensorflow::AttrValue::kI: {
      std::string tmp_tostr;
      all2str(tmp_tostr, attr_value.i());
      std::cout << "   (kInt) " << attr_string << ": " << tmp_tostr << std::endl;
    }
      break;
    case tensorflow::AttrValue::kF: {
      std::string tmp_tostr;
      all2str(tmp_tostr, attr_value.f());
      std::cout << "   (kFloat) " << attr_string << ": " << tmp_tostr << std::endl;
    }
      break;
    case tensorflow::AttrValue::kB: {
      std::string tmp_tostr;
      all2str(tmp_tostr, attr_value.b());
      std::cout << "   (kBool) " << attr_string << ": " << tmp_tostr << std::endl;
    }
      break;
    case tensorflow::AttrValue::kType: {
      std::string tmp_tostr;
      all2str(tmp_tostr, attr_value.type());
      std::cout << "   (kType) " << attr_string << ": " << tmp_tostr << std::endl;
    }
      break;
    case tensorflow::AttrValue::kShape: {
      std::string tmp_tostr;
      std::cout << "   (kShape) " << attr_string << ": ";
      if (attr_value.shape().unknown_rank()) {
        std::cout << "<unknown>";
      }
      std::string s = "[";
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
      std::cout << s << std::endl;
    }
      break;
    case tensorflow::AttrValue::kTensor: {
      std::string tmp_tostr;
      /* Both way is a TensorFlow representation of all Tensor information
      std::cout << "   (kTensor) " << attr_string << ": " << attr_value.tensor().ShortDebugString();
      std::cout << "   (kTensor) " << attr_string << ": " << attr_value.tensor().DebugString();
      */
      if (attr_value.tensor().dtype() == tensorflow::DT_FLOAT) {
        std::cout << "   (kTensor) " << attr_string << ": [";
        for (int i = 0; i < attr_value.tensor().tensor_content().size()/4; i++)
          std::cout << char_to_float(attr_value.tensor().tensor_content().c_str() + 4*i) << ' ';
        std::cout << ']' << std::endl;
      } else if (attr_value.tensor().dtype() == tensorflow::DT_INT32){
        std::cout << "   (kTensor) " << attr_string << ": [";
        for (int i = 0; i < attr_value.tensor().tensor_content().size()/4; i++)
          std::cout << char_to_int32(attr_value.tensor().tensor_content().c_str() + 4*i) << ' ';
        std::cout << ']' << std::endl;

      }
      std::cout << "   kTensor Size: " << attr_string << ": ";
      for (const auto& d : attr_value.tensor().tensor_shape().dim()) {
        if (d.size() == -1)
          std::cout << "What? Size is undefined here";
        else {
          all2str(tmp_tostr, d.size());
          std::cout << ' ' << tmp_tostr;
        }
      }
      std::cout << std::endl;
      break;
    }
    case tensorflow::AttrValue::kList: {
      std::string tmp_tostr;
      std::cout << "   (kList) " << attr_string << ": ";
      std::string ret = "[";
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
      std::cout << ret << std::endl;
      break;
    }
    case tensorflow::AttrValue::kFunc: {
      std::string tmp_tostr;
      for (auto p : attr_value.func().attr()) {
        std::cout << "   SubAttr: " << p.first << ": " << std::endl;
        summarize_attr_value(p.first, p.second);
      }
      break;
    }
    case tensorflow::AttrValue::kPlaceholder: {
      std::string tmp_tostr;
      all2str(tmp_tostr, attr_value.placeholder());
      std::cout << "   (kPlaceholder) " << attr_string << ": " << tmp_tostr << std::endl;
    }
      break;
    case tensorflow::AttrValue::VALUE_NOT_SET: {
      std::cout << "   <Unknown AttrValue type> " << std::endl;
    }
      break;
  }
}

inline bool is_placeholder(const tensorflow::NodeDef& node_def) {
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

inline bool is_scalar_const(const tensorflow::NodeDef& node_def) {
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
inline void list_nodes(const tensorflow::GraphDef& graph_def) {
  std::cout << "  Node Size: " << graph_def.node_size() << std::endl;
  for (int i = 0; i < graph_def.node_size(); i++) {
    const tensorflow::NodeDef& node_def = graph_def.node(i);
    // print nodes information
    fprintf(stderr, "\nNode: %s (%s)\n", node_def.name().c_str(), node_def.op().c_str());
    bool print_simple_check = false;
    bool print_inputs = true;

    // print a simple check on placeholder and const nodes
    if (print_simple_check) {
      std::cout << "Is this const or placeholder: "
          << static_cast<bool>(is_scalar_const(node_def) && is_placeholder(node_def))
          << std::endl;
    }

    // print the inputs of nodes
    if (print_inputs) {
      std::cout << "  Inputs: ";
      for (const std::string& input : node_def.input()) {
        std::cout << input << ' ';
      }
      std::cout << std::endl;
    }

    // print attribution of nodes in TensorFlow
    std::cout << " Attrbutes are shown below:" << std::endl;
    for (const auto& attr : node_def.attr()) {
      auto iter = node_def.attr().find(attr.first);
      summarize_attr_value(attr.first, iter->second);
    }
  }
}
}