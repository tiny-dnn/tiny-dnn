#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include "graph.pb.h"
using namespace std;
using namespace tensorflow;

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