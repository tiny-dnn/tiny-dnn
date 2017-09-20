/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#include <iostream>

#include "tiny_dnn/tiny_dnn.h"

///////////////////////////////////////////////////////////////////////////////
// recongnition on MNIST similar to LaNet-5 adding deconvolution

void deconv_lanet(tiny_dnn::network<tiny_dnn::graph> &nn,
                  std::vector<tiny_dnn::label_t> train_labels,
                  std::vector<tiny_dnn::label_t> test_labels,
                  std::vector<tiny_dnn::vec_t> train_images,
                  std::vector<tiny_dnn::vec_t> test_images) {
// connection table [Y.Lecun, 1998 Table.1]
#define O true
#define X false
  // clang-format off
static const bool tbl[] = {
    O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
    O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
    O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
    X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
    X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
    X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
};
// clang-format on
#undef O
#undef X

  // declare nodes
  tiny_dnn::input_layer i1(tiny_dnn::shape3d(32, 32, 1));
  tiny_dnn::convolutional_layer c1(32, 32, 5, 1, 6);
  tiny_dnn::tanh_layer c1_tanh(28, 28, 6);
  tiny_dnn::average_pooling_layer p1(28, 28, 6, 2);
  tiny_dnn::tanh_layer p1_tanh(14, 14, 6);
  tiny_dnn::deconvolutional_layer d1(
    14, 14, 5, 6, 16, tiny_dnn::core::connection_table(tbl, 6, 16));
  tiny_dnn::tanh_layer d1_tanh(18, 18, 16);
  tiny_dnn::average_pooling_layer p2(18, 18, 16, 2);
  tiny_dnn::tanh_layer p2_tanh(9, 9, 16);
  tiny_dnn::convolutional_layer c2(9, 9, 9, 16, 120);
  tiny_dnn::tanh_layer c2_tanh(1, 1, 120);
  tiny_dnn::fully_connected_layer fc1(120, 10);
  tiny_dnn::tanh_layer fc1_tanh(10);

  // connecting activation layers behind other layers
  c1 << c1_tanh;
  p1 << p1_tanh;
  d1 << d1_tanh;
  c2 << c2_tanh;
  p2 << p2_tanh;
  fc1 << fc1_tanh;

  // connect them to graph
  i1 << c1 << p1 << d1 << p2 << c2 << fc1;
  construct_graph(nn, {&i1}, {&fc1});

  std::cout << "start training" << std::endl;

  tiny_dnn::progress_display disp(train_images.size());
  tiny_dnn::timer t;
  int minibatch_size = 10;
  int num_epochs     = 30;

  tiny_dnn::adagrad optimizer;
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

  // save networks
  std::ofstream ofs("deconv_lanet_weights");
  ofs << nn;
}

///////////////////////////////////////////////////////////////////////////////
// Deconcolutional Auto-encoder
void deconv_ae(tiny_dnn::network<tiny_dnn::sequential> &nn,
               std::vector<tiny_dnn::label_t> train_labels,
               std::vector<tiny_dnn::label_t> test_labels,
               std::vector<tiny_dnn::vec_t> train_images,
               std::vector<tiny_dnn::vec_t> test_images) {
  // construct nets
  nn << tiny_dnn::convolutional_layer(32, 32, 5, 1, 6)
     << tiny_dnn::tanh_layer(28, 28, 6)
     << tiny_dnn::average_pooling_layer(28, 28, 6, 2)
     << tiny_dnn::tanh_layer(14, 14, 6)
     << tiny_dnn::convolutional_layer(14, 14, 3, 6, 16)
     << tiny_dnn::tanh_layer(12, 12, 16)
     << tiny_dnn::deconvolutional_layer(12, 12, 3, 16, 6)
     << tiny_dnn::tanh_layer(14, 14, 6)
     << tiny_dnn::average_unpooling_layer(14, 14, 6, 2)
     << tiny_dnn::tanh_layer(28, 28, 6)
     << tiny_dnn::deconvolutional_layer(28, 28, 5, 6, 1)
     << tiny_dnn::tanh_layer(32, 32, 6);

  // load train-data and make corruption

  std::vector<tiny_dnn::vec_t> training_images_corrupted(train_images);

  for (auto &d : training_images_corrupted) {
    d = corrupt(std::move(d), tiny_dnn::float_t(0.1),
                tiny_dnn::float_t(0.0));  // corrupt 10% data
  }

  tiny_dnn::gradient_descent optimizer;

  // learning deconcolutional Auto-encoder
  nn.train<tiny_dnn::mse>(optimizer, training_images_corrupted, train_images);

  std::cout << "end training." << std::endl;

  // save networks
  std::ofstream ofs("deconv_ae_weights");
  ofs << nn;
}

///////////////////////////////////////////////////////////////////////////////
// ENet
void enet(tiny_dnn::network<tiny_dnn::graph> &nn,
          std::vector<tiny_dnn::label_t> train_labels,
          std::vector<tiny_dnn::label_t> test_labels,
          std::vector<tiny_dnn::vec_t> train_images,
          std::vector<tiny_dnn::vec_t> test_images) {
  // initial module
  tiny_dnn::input_layer ii0(tiny_dnn::shape3d(32, 32, 1));
  tiny_dnn::convolutional_layer ic1(32, 32, 3, 1, 8, tiny_dnn::padding::same,
                                    true, 2, 2);
  tiny_dnn::tanh_layer ic1_tanh(16, 16, 8);
  tiny_dnn::max_pooling_layer ip1(32, 32, 1, 2);
  tiny_dnn::tanh_layer ip1_tanh(16, 16, 1);
  tiny_dnn::convolutional_layer ic2(16, 16, 1, 1, 8, tiny_dnn::padding::same);
  tiny_dnn::tanh_layer ic2_tanh(16, 16, 8);
  tiny_dnn::concat_layer icc1(2, 16 * 16 * 8);

  // connecting activation layers behind other layers
  ic1 << ic1_tanh;
  ic2 << ic2_tanh;
  ip1 << ip1_tanh;

  ii0 << ip1 << ic2;
  ii0 << ic1;
  (ic2, ic1) << icc1;

  // bottle neck module 1
  tiny_dnn::max_pooling_layer b1p1(16, 16, 16, 2);
  tiny_dnn::tanh_layer b1p1_tanh(8, 8, 16);
  tiny_dnn::convolutional_layer b1c2(8, 8, 1, 16, 32, tiny_dnn::padding::same);
  tiny_dnn::tanh_layer b1c2_tanh(8, 8, 32);
  tiny_dnn::convolutional_layer b1c1(16, 16, 1, 16, 32,
                                     tiny_dnn::padding::same);
  tiny_dnn::tanh_layer b1c1_tanh(16, 16, 32);
  tiny_dnn::convolutional_layer b1c3(16, 16, 2, 32, 32, tiny_dnn::padding::same,
                                     true, 2, 2);
  tiny_dnn::tanh_layer b1c3_tanh(8, 8, 32);
  tiny_dnn::convolutional_layer b1c4(8, 8, 1, 32, 32, tiny_dnn::padding::same);
  tiny_dnn::tanh_layer b1c4_tanh(8, 8, 32);
  tiny_dnn::concat_layer b1cc1(2, 8 * 8 * 32);

  // connecting activation layers behind other layers
  b1p1 << b1p1_tanh;
  b1c1 << b1c1_tanh;
  b1c2 << b1c2_tanh;
  b1c3 << b1c3_tanh;
  b1c4 << b1c4_tanh;

  icc1 << b1p1 << b1c2;
  icc1 << b1c1 << b1c3 << b1c4;
  (b1c2, b1c4) << b1cc1;

  // bottle neck module 2
  tiny_dnn::deconvolutional_layer b2d1(8, 8, 1, 64, 16, tiny_dnn::padding::same,
                                       true, 2, 2);
  tiny_dnn::tanh_layer b2d1_tanh(8, 8, 16);
  tiny_dnn::deconvolutional_layer b2d2(16, 16, 1, 16, 1,
                                       tiny_dnn::padding::same, true, 2, 2);
  tiny_dnn::tanh_layer b2d2_tanh(16, 16, 1);

  tiny_dnn::fully_connected_layer fc1(32 * 32, 10);
  tiny_dnn::tanh_layer fc1_tanh(10);

  // connecting activation layers behind deconv and fc layers
  b2d1 << b2d1_tanh;
  b2d2 << b2d2_tanh;
  fc1 << fc1_tanh;

  b1cc1 << b2d1 << b2d2 << fc1;

  // construct whole network
  construct_graph(nn, {&ii0}, {&fc1});

  // load train-data and make corruption

  std::cout << "start training" << std::endl;

  tiny_dnn::progress_display disp(train_images.size());
  tiny_dnn::timer t;
  int minibatch_size = 10;
  int num_epochs     = 30;

  tiny_dnn::adagrad optimizer;
  optimizer.alpha *= tiny_dnn::float_t(std::sqrt(minibatch_size));

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

  // save networks
  std::ofstream ofs("deconv_lanet_weights");
  ofs << nn;
}

void train(std::string data_dir_path, std::string experiment) {
  std::cout << "load traing and testing data..." << std::endl;

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
  // specify loss-function and learning strategy
  tiny_dnn::network<tiny_dnn::sequential> nn_s;
  tiny_dnn::network<tiny_dnn::graph> nn_g;

  if (experiment == "deconv_lanet")
    deconv_lanet(nn_g, train_labels, test_labels, train_images,
                 test_images);  // recongnition on MNIST similar to LaNet-5
  // adding deconvolution
  else if (experiment == "deconv_ae")
    deconv_ae(nn_s, train_labels, test_labels, train_images,
              test_images);  // Deconcolution Auto-encoder on MNIST
  else if (experiment == "enet")
    enet(nn_g, train_labels, test_labels, train_images,
         test_images);  // Bottle neck module based ENet
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage : " << argv[0]
              << " path_to_data (example:../data) (example:deconv_lanet, "
                 "deconv_ae or enet)"
              << std::endl;
    return -1;
  }
  train(argv[1], argv[2]);
}
