#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include "tiny_dnn/io/tensorflow/proto_parser.h"
using namespace std;
using namespace tiny_cnn;
using namespace tensorflow;

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

  list_nodes(graph_def);

  // Optional:  Delete all global objects allocated by libprotobuf.
  google::protobuf::ShutdownProtobufLibrary();

  return 0;
}