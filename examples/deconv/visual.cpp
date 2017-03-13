/*
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#include <iostream>
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace std;

// rescale output to 0-100
template <typename Activation>
double rescale(double x) {
  Activation a(1);
  return 100.0 * (x - a.scale().first) / (a.scale().second - a.scale().first);
}

void convert_image(const std::string &imagefilename,
                   double minv,
                   double maxv,
                   int w,
                   int h,
                   vec_t &data) {
  image<> img(imagefilename, image_type::grayscale);
  image<> resized = resize_image(img, w, h);

  // mnist dataset is "white on black", so negate required
  std::transform(
    resized.begin(), resized.end(), std::back_inserter(data),
    [=](uint8_t c) { return (255 - c) * (maxv - minv) / 255.0 + minv; });
}

void construct_net(network<sequential> &nn) {
  // construct nets
  nn << convolutional_layer(32, 32, 5, 1, 6) << tanh_layer(28, 28, 6)
     << average_pooling_layer(28, 28, 6, 2)
     << convolutional_layer(14, 14, 5, 6, 16) << tanh_layer(10, 10, 16)
     << deconvolutional_layer(10, 10, 5, 16, 6) << tanh_layer(14, 14, 6)
     << average_unpooling_layer(14, 14, 6, 2)
     << deconvolutional_layer(28, 28, 5, 6, 1) << tanh_layer(32, 32, 1);
}

void train_network(network<sequential> nn, const string &train_dir_path) {
  // load train-data and make corruption
  // load MNIST dataset
  std::cout << "load traing and testing data..." << std::endl;
  std::vector<label_t> train_labels, test_labels;
  std::vector<vec_t> train_images, test_images;

  parse_mnist_labels(train_dir_path + "/train-labels.idx1-ubyte",
                     &train_labels);
  parse_mnist_images(train_dir_path + "/train-images.idx3-ubyte", &train_images,
                     -1.0, 1.0, 2, 2);
  parse_mnist_labels(train_dir_path + "/t10k-labels.idx1-ubyte", &test_labels);
  parse_mnist_images(train_dir_path + "/t10k-images.idx3-ubyte", &test_images,
                     -1.0, 1.0, 2, 2);
  std::vector<vec_t> training_images_corrupted(train_images);

  for (auto &d : training_images_corrupted) {
    d = corrupt(move(d), 0.1f, 0.0f);  // corrupt 10% data
  }

  gradient_descent optimizer;

  std::cout << "start training deconvolutional auto-encoder..." << std::endl;

  // learning deconcolutional Auto-encoder
  nn.train<mse>(optimizer, training_images_corrupted, train_images);

  std::cout << "end training." << std::endl;

  // save networks
  std::ofstream ofs("deconv_ae_weights");
  ofs << nn;
}

void recognize(const std::string &dictionary,
               const std::string &src_filename,
               const string &train_dir_path = "") {
  network<sequential> nn;

  construct_net(nn);
  // training
  if (train_dir_path != "")
    train_network(nn, train_dir_path);
  else
    cout << "make sure you have already got a trained model" << std::endl;

  // load nets
  ifstream ifs(dictionary.c_str());
  ifs >> nn;

  // convert imagefile to vec_t
  vec_t data;
  convert_image(src_filename, -1.0, 1.0, 32, 32, data);

  std::cout << "start predicting on single image..." << std::endl;

  // recognize
  auto res = nn.predict(data);
  vector<pair<double, int>> scores;

  // sort & print top-3
  for (int i = 0; i < 10; i++)
    scores.emplace_back(rescale<tanh_layer>(res[i]), i);

  sort(scores.begin(), scores.end(), greater<pair<double, int>>());

  for (int i = 0; i < 3; i++)
    cout << scores[i].second << "," << scores[i].first << endl;

  // visualize outputs of each layer
  for (size_t i = 0; i < nn.layer_size(); i++) {
    auto out_img  = nn[i]->output_to_image();
    auto filename = "layer_" + std::to_string(i) + ".png";
    out_img.save(filename);
  }
  // visualize filter shape of first convolutional layer
  auto weightc = nn.at<convolutional_layer>(0).weight_to_image();
  weightc.save("weights.png");
}

int main(int argc, char **argv) {
  if (argc < 2) {
    cout << "please specify training data path and testing image file"
         << std::endl;
    return 0;
  } else if (argc == 2) {
    recognize("deconv_ae_weights", argv[1]);
  } else
    recognize("deconv_ae_weights", argv[1], argv[2]);
}
