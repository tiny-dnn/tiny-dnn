#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include "tiny_cnn/tiny_cnn.h"
#include "graph.pb.h"
#include "node_factory.h"
#include "node_factory_impl.h"
using namespace std;
using namespace tensorflow;
using namespace tiny_cnn;
using namespace tiny_cnn::activation;

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

  auto net = create_net_from_tensorflow_prototxt(graph_def);
  // reload_weight_from_tensorflow_protobinary(trained_file, net.get());
  // Optional:  Delete all global objects allocated by libprotobuf.
  google::protobuf::ShutdownProtobufLibrary();

  return 0;
}