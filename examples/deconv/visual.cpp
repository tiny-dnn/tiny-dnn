/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#include <iostream>

#include "tiny_dnn/tiny_dnn.h"

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
                   tiny_dnn::vec_t &data) {
  tiny_dnn::image<> img(imagefilename, tiny_dnn::image_type::grayscale);
  tiny_dnn::image<> resized = resize_image(img, w, h);

  // mnist dataset is "white on black", so negate required
  std::transform(
    resized.begin(), resized.end(), std::back_inserter(data),
    [=](uint8_t c) { return (255 - c) * (maxv - minv) / 255.0 + minv; });
}

void construct_net(tiny_dnn::network<tiny_dnn::sequential> &nn) {
  // construct nets
  nn << tiny_dnn::convolutional_layer(32, 32, 5, 1, 6)
     << tiny_dnn::tanh_layer(28, 28, 6)
     << tiny_dnn::average_pooling_layer(28, 28, 6, 2)
     << tiny_dnn::convolutional_layer(14, 14, 5, 6, 16)
     << tiny_dnn::tanh_layer(10, 10, 16)
     << tiny_dnn::deconvolutional_layer(10, 10, 5, 16, 6)
     << tiny_dnn::tanh_layer(14, 14, 6)
     << tiny_dnn::average_unpooling_layer(14, 14, 6, 2)
     << tiny_dnn::deconvolutional_layer(28, 28, 5, 6, 1)
     << tiny_dnn::tanh_layer(32, 32, 1);
}

void train_network(tiny_dnn::network<tiny_dnn::sequential> nn,
                   const std::string &train_dir_path) {
  // load train-data and make corruption
  // load MNIST dataset
  std::cout << "load traing and testing data..." << std::endl;
  std::vector<tiny_dnn::label_t> train_labels, test_labels;
  std::vector<tiny_dnn::vec_t> train_images, test_images;

  tiny_dnn::parse_mnist_labels(train_dir_path + "/train-labels.idx1-ubyte",
                               &train_labels);
  tiny_dnn::parse_mnist_images(train_dir_path + "/train-images.idx3-ubyte",
                               &train_images, -1.0, 1.0, 2, 2);
  tiny_dnn::parse_mnist_labels(train_dir_path + "/t10k-labels.idx1-ubyte",
                               &test_labels);
  tiny_dnn::parse_mnist_images(train_dir_path + "/t10k-images.idx3-ubyte",
                               &test_images, -1.0, 1.0, 2, 2);
  std::vector<tiny_dnn::vec_t> training_images_corrupted(train_images);

  for (auto &d : training_images_corrupted) {
    d = corrupt(move(d), 0.1f, 0.0f);  // corrupt 10% data
  }

  tiny_dnn::gradient_descent optimizer;

  std::cout << "start training deconvolutional auto-encoder..." << std::endl;

  // learning deconcolutional Auto-encoder
  nn.train<tiny_dnn::mse>(optimizer, training_images_corrupted, train_images);

  std::cout << "end training." << std::endl;

  // save networks
  std::ofstream ofs("deconv_ae_weights");
  ofs << nn;
}

void recognize(const std::string &dictionary,
               const std::string &src_filename,
               const std::string &train_dir_path = "") {
  tiny_dnn::network<tiny_dnn::sequential> nn;

  construct_net(nn);
  // training
  if (train_dir_path != "")
    train_network(nn, train_dir_path);
  else
    std::cout << "make sure you have already got a trained model" << std::endl;

  // load nets
  std::ifstream ifs(dictionary.c_str());
  ifs >> nn;

  // convert imagefile to vec_t
  tiny_dnn::vec_t data;
  convert_image(src_filename, -1.0, 1.0, 32, 32, data);

  std::cout << "start predicting on single image..." << std::endl;

  // recognize
  auto res = nn.predict(data);
  std::vector<std::pair<double, int>> scores;

  // sort & print top-3
  for (int i = 0; i < 10; i++)
    scores.emplace_back(rescale<tiny_dnn::tanh_layer>(res[i]), i);

  sort(scores.begin(), scores.end(), std::greater<std::pair<double, int>>());

  for (int i = 0; i < 3; i++) {
    std::cout << scores[i].second << "," << scores[i].first << std::endl;
  }

  // visualize outputs of each layer
  for (size_t i = 0; i < nn.layer_size(); i++) {
    auto out_img  = nn[i]->output_to_image();
    auto filename = "layer_" + std::to_string(i) + ".png";
    out_img.save(filename);
  }
  // visualize filter shape of first convolutional layer
  auto weightc = nn.at<tiny_dnn::convolutional_layer>(0).weight_to_image();
  weightc.save("weights.png");
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "please specify training data path and testing image file"
              << std::endl;
    return 0;
  } else if (argc == 2) {
    recognize("deconv_ae_weights", argv[1]);
  } else {
    recognize("deconv_ae_weights", argv[1], argv[2]);
  }
}
