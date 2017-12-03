/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#include <iostream>

#include "tiny_dnn/tiny_dnn.h"

void construct_net(tiny_dnn::network<tiny_dnn::sequential> &nn) {
// connection table [Y.Lecun, 1998 Table.1]
#define O true
#define X false
  // clang-format off
  static const bool tbl[] = {
    O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O, O, X, X, X, O, O, O,
    X, X, O, O, O, O, X, O, O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
    X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O, X, X, O, O, O, X, X, O,
    O, O, O, X, O, O, X, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O};
#undef O
#undef X

  using q_conv = tiny_dnn::quantized_convolutional_layer;
  using ave_pool = tiny_dnn::average_pooling_layer;
  using q_fc = tiny_dnn::quantized_fully_connected_layer;

  tiny_dnn::core::backend_t backend_type = tiny_dnn::core::backend_t::internal;
  // construct nets
  //
  // C : convolution
  // S : sub-sampling
  // F : fully connected
  nn << q_conv(32, 32, 5, 1, 6, tiny_dnn::padding::valid, true, 1, 1,
          backend_type)                    // C1, 1@32x32-in, 6@28x28-out
     << tiny_dnn::tanh_layer(28, 28, 6)
     << ave_pool(28, 28, 6, 2)             // S2, 6@28x28-in, 6@14x14-out
     << tiny_dnn::tanh_layer(14, 14, 6)
     << q_conv(14, 14, 5, 6, 16, tiny_dnn::core::connection_table(tbl, 6, 16),
               tiny_dnn::padding::valid, true, 1, 1,
               backend_type)               // C3, 6@14x14-in, 16@10x10-in
     << tiny_dnn::tanh_layer(10, 10, 16)
     << ave_pool(10, 10, 16, 2)            // S4, 16@10x10-in, 16@5x5-out
     << tiny_dnn::tanh_layer(5, 5, 16)
     << q_conv(5, 5, 5, 16, 120, tiny_dnn::padding::valid, true, 1, 1,
               backend_type)               // C5, 16@5x5-in, 120@1x1-out
     << tiny_dnn::tanh_layer(120)
     << q_fc(120, 10, true, backend_type)
  // F6, 120-in, 10-out
     << tiny_dnn::tanh_layer(10);
  // clang-format on
}

static void train_lenet(const std::string &data_dir_path) {
  // specify loss-function and learning strategy
  tiny_dnn::network<tiny_dnn::sequential> nn;
  tiny_dnn::adagrad optimizer;

  construct_net(nn);

  std::cout << "load models..." << std::endl;

  // load MNIST dataset
  std::vector<tiny_dnn::label_t> train_labels, test_labels;
  std::vector<tiny_dnn::vec_t> train_images, test_images;

  tiny_dnn::parse_mnist_labels(data_dir_path + "/train-labels.idx1-ubyte",
                               &train_labels);
  tiny_dnn::parse_mnist_images(data_dir_path + "/train-images.idx3-ubyte",
                               &train_images, -1.0, 1.0, 2, 2);
  tiny_dnn::parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte",
                               &test_labels);
  tiny_dnn::parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte",
                               &test_images, -1.0, 1.0, 2, 2);

  std::cout << "start training" << std::endl;

  tiny_dnn::progress_display disp(train_images.size());
  tiny_dnn::timer t;
  int minibatch_size = 10;
  int num_epochs     = 30;

  optimizer.alpha *= static_cast<tiny_dnn::float_t>(std::sqrt(minibatch_size));

  // create callback
  auto on_enumerate_epoch = [&]() {
    std::cout << t.elapsed() << "s elapsed." << std::endl;
    tiny_dnn::result res = nn.test(test_images, test_labels);
    std::cout << res.num_success << "/" << res.num_total << std::endl;

    disp.restart(train_images.size());
    t.restart();
  };

  auto on_enumerate_minibatch = [&]() { disp += minibatch_size; };

  // training
  nn.train<tiny_dnn::mse>(optimizer, train_images, train_labels, minibatch_size,
                          num_epochs, on_enumerate_minibatch,
                          on_enumerate_epoch);

  std::cout << "end training." << std::endl;

  // test and show results
  nn.test(test_images, test_labels).print_detail(std::cout);

  // save network model & trained weights
  nn.save("LeNet-model");
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage : " << argv[0] << " path_to_data (example:../data)"
              << std::endl;
    return -1;
  }
  train_lenet(argv[1]);
  return 0;
}
