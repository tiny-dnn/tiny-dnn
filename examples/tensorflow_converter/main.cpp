#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include "graph.pb.h"
using namespace std;
using namespace tensorflow;

void summarize_attr_value(const string& attr_string, const AttrValue& attr_value) {
  switch (attr_value.value_case()) {
    case AttrValue::kS:
      cout << "  kString: " << attr_string << ": " << attr_value.s() << endl;
    case AttrValue::kI:
      cout << "  kInt: " << attr_string << ": " << attr_value.i() << endl;
    case AttrValue::kF:
      cout << "  kFloat: " << attr_string << ": " << attr_value.f() << endl;
    case AttrValue::kB:
      cout << "  kBool: " << attr_string << ": " << attr_value.b() << endl;
    case AttrValue::kType:
      cout << "  kType: " << attr_string << ": " << attr_value.type() << endl;
    case AttrValue::kShape: {
      cout << "  kShape: " << attr_string << ": ";
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
          s += d.size();
        }
        first = false;
      }
      s += "]";
      cout << s << endl;
    }
    case AttrValue::kTensor: {
      cout << "  kTensor: " << attr_string << ": " << attr_value.tensor().ShortDebugString();
      // cout << "  kTensor: " << attr_string << ": " << attr_value.tensor().DebugString();
      // cout << "  kTensor: " << attr_string << ": " << atoi(attr_value.tensor().tensor_content().c_str());
      cout << "  kTensor Size: " << attr_string << ": ";
      for (const auto& d : attr_value.tensor().tensor_shape().dim()) {
        if (d.size() == -1)
          cout << "What? Size is undefined here";
        else
          cout << ' ' << d.size();
      }
      cout << endl;
    }
    case AttrValue::kList: {
      cout << "  kList: " << attr_string << ": ";
      string ret = "[";
      if (attr_value.list().s_size() > 0) {
        for (int i = 0; i < attr_value.list().s_size(); ++i) {
          if (i > 0) ret += ", ";
          ret += attr_value.list().s(i);
        }
      } else if (attr_value.list().i_size() > 0) {
        for (int i = 0; i < attr_value.list().i_size(); ++i) {
          if (i > 0) ret += ", ";
          ret += attr_value.list().i(i);
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
              ret += "What? Size is undefined here";
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
    }
    case AttrValue::kFunc: {
      for (auto p : attr_value.func().attr()) {
        cout << "  SubAttr: " << p.first << ": " << endl;
        summarize_attr_value(p.first, p.second);
      }
    }
    case AttrValue::kPlaceholder:
      cout << "  kPlaceholder: " << attr_string << ": " << attr_value.placeholder() << endl;
    case AttrValue::VALUE_NOT_SET:
      cout << "  <Unknown AttrValue type> " << endl;
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
    fprintf(stderr, "Node: %s (%s)\n", node_def.name().c_str(), node_def.op().c_str());
    fprintf(stderr, "  Is Placeholder: %d, Is Const: %d\n", IsPlaceholder(node_def), IsScalarConst(node_def));

    cout << "  Inputs: ";
    for (const string& input : node_def.input()) {
      cout << input << ' ';
    }
    cout << endl;
    std::vector<string> attr_names;
    attr_names.reserve(node_def.attr().size());
    for (const auto& attr : node_def.attr()) {
      attr_names.push_back(attr.first);
      auto iter = node_def.attr().find(attr.first);
      cout << " Attrbutes are shown below:" << endl;
      summarize_attr_value(attr.first, iter->second);
    }
         /*
    if (node_def.op() != "Placeholder" && node_def.op() != "Identity" && node_def.op() == "Conv2D") {
      cout  << "  Tensor type: " << node_def.attr().at("dtype").tensor().dtype()
            << "  Size of tensor values: " << node_def.attr().at("value").tensor().float_val().size()
            << endl;

      std::vector<float> parameters;
      for (int i = 0; i < node_def.attr().at("value").tensor().float_val().size(); i++) {
        parameters.push_back(node_def.attr().at("value").tensor().float_val().Get(i));
      }
    }*/
  }
}

// Main function:  Reads the graph definition from a file and prints all
// the information inside.
int main(int argc, char* argv[]) {
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  if (argc != 2) {
    cerr << "Usage:  " << argv[0] << " GRAPH_DEF_FILE" << endl;
    return -1;
  }

  tensorflow::GraphDef graph_def;

  {
    // Read the existing graph.
    fstream input(argv[1], ios::in | ios::binary);
    if (!graph_def.ParseFromIstream(&input)) {
      cerr << "Failed to parse graph." << endl;
      return -1;
    }
  }

  ListNodes(graph_def);

  // Optional:  Delete all global objects allocated by libprotobuf.
  google::protobuf::ShutdownProtobufLibrary();

  return 0;
}