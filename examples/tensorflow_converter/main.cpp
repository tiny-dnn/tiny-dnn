#include <iostream>
#include <fstream>
#include <string>
#include "graph.pb.h"
using namespace std;

// Iterates though all people in the GraphDef and prints info about them.
void ListNodes(const tensorflow::GraphDef& graph_def) {
  for (int i = 0; i < graph_def.node_size(); i++) {
    const tensorflow::NodeDef& node_def = graph_def.node(i);

    cout << "  Node_Def Name: " << node_def.name() << endl;
    cout << "  Node_Def Operation: " << node_def.op() << endl;
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