/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.
    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#include <ctime>
#include <iostream>
#include <memory>
#define NO_STRICT
#define CNN_USE_CAFFE_CONVERTER
#define DNN_USE_IMAGE_API
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace std;

void convert(const string& model_file, const string& trained_file) {
  // load model and weights
  auto net = create_net_from_caffe_prototxt(model_file);
  reload_weight_from_caffe_protobinary(trained_file, net.get());

  // save model, weights, architecture
  net->save("tiny-model");  // Saves both the architecture and weights
  // net->save("tiny-weights-json", content_type::weights, file_format::json);
  // net->save("tiny-weights-binary", content_type::weights,
  // file_format::binary);
  // net->save("tiny-architecture", content_type::model, file_format::json);
}

int main(int argc, char** argv) {
  if (argc != 3) {
    cout << "Usage: " << argv[0] << " <Prototxt file> <Caffemodel file>"
         << endl;
    return 0;
  }
  try {
    convert(argv[1], argv[2]);
  } catch (const nn_error& e) {
    cout << e.what() << endl;
  }
}
