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

void recognize(const std::string &dictionary, const std::string &src_filename) {
  tiny_dnn::network<tiny_dnn::sequential> nn;

  nn.load(dictionary);

  // convert imagefile to vec_t
  tiny_dnn::vec_t data;
  convert_image(src_filename, -1.0, 1.0, 32, 32, data);

  // recognize
  auto res = nn.predict(data);
  std::vector<std::pair<double, int>> scores;

  // sort & print top-3
  for (int i = 0; i < 10; i++)
    scores.emplace_back(rescale<tiny_dnn::tanh_layer>(res[i]), i);

  sort(scores.begin(), scores.end(), std::greater<std::pair<double, int>>());

  for (int i = 0; i < 3; i++)
    std::cout << scores[i].second << "," << scores[i].first << std::endl;

  // save outputs of each layer
  for (size_t i = 0; i < nn.depth(); i++) {
    auto out_img  = nn[i]->output_to_image();
    auto filename = "layer_" + std::to_string(i) + ".png";
    out_img.save(filename);
  }
  // save filter shape of first convolutional layer
  {
    auto weight   = nn.at<tiny_dnn::convolutional_layer>(0).weight_to_image();
    auto filename = "weights.png";
    weight.save(filename);
  }
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "please specify image file" << std::endl;
    return 0;
  }
  recognize("LeNet-model", argv[1]);
}
