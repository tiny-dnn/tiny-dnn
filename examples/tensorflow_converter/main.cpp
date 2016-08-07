#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include "graph.pb.h"
using namespace std;

// Iterates though all nodes in the GraphDef and prints info about them.
void ListNodes(const tensorflow::GraphDef& graph_def) {
  cout << "  Node Size: " << graph_def.node_size() << endl;
  for (int i = 0; i < graph_def.node_size(); i++) {
    const tensorflow::NodeDef& node_def = graph_def.node(i);
    // print nodes information
    cout << "  Node name: " << node_def.name()
         << "  Node operation: " << node_def.op()
         << endl;
    if (node_def.op() != "Placeholder" && node_def.op() != "Identity") {
      cout << "  Tensor type: " << node_def.attr().at("dtype").tensor().dtype()
           << "  Size of tensor values: " << node_def.attr().at("value").tensor().float_val().size()
           << endl;

      std::vector<float> parameters;
      for (int i = 0; i < node_def.attr().at("value").tensor().float_val().size(); i++) {
        parameters.push_back(node_def.attr().at("value").tensor().float_val().Get(i));
      }
    }
  }
}

// Main function:  Reads the graph definition from a file and prints all
//   the information inside.
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