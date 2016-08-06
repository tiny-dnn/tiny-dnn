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
    cout << "  Node Name: " << node_def.name()
         << "  Node Operation: " << node_def.op()
         << endl;

    // print attributes information
    google::protobuf::Map<string, tensorflow::AttrValue>::const_iterator iMapPairLocator;

    for ( iMapPairLocator = node_def.attr().begin()
        ; iMapPairLocator != node_def.attr().end()
        ; ++ iMapPairLocator )
    {
        cout << "  Key: " << static_cast<string>(iMapPairLocator->first.c_str())
             << endl;
    }
    cout << "  TensorShapeSize: " << node_def.attr().at("dtype").tensor().tensor_shape().dim_size()
         << "  TensorContent: " << node_def.attr().at("value").tensor().tensor_shape().dim_size()
         << endl;
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