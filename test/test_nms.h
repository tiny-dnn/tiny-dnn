/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <vector>
#define IMAGE_SIZE 300

namespace tiny_dnn {

TEST(nms, nms_check) {
  std::vector<bounding_box> bboxes;

  // clang-format off
  tiny_dnn::bounding_box b;
  b.x_min = IMAGE_SIZE * 0.0275; b.y_min = IMAGE_SIZE * 0.2787;
  b.x_max = IMAGE_SIZE * 0.8972; b.y_max = IMAGE_SIZE * 0.7751;
  b.score = 0.6042;
  bboxes.push_back(b);
  b.x_min = IMAGE_SIZE * 0.0325; b.y_min = IMAGE_SIZE * 0.3053;
  b.x_max = IMAGE_SIZE * 0.8765; b.y_max = IMAGE_SIZE * 0.7529;
  b.score = 0.5606;
  bboxes.push_back(b);
  b.x_min = IMAGE_SIZE * 0.0310; b.y_min = IMAGE_SIZE * 0.2693;
  b.x_max = IMAGE_SIZE * 0.9720; b.y_max = IMAGE_SIZE * 0.7624;
  b.score = 0.6192;
  bboxes.push_back(b);
  b.x_min = IMAGE_SIZE * 0.0288; b.y_min = IMAGE_SIZE * 0.2536;
  b.x_max = IMAGE_SIZE * 0.9700; b.y_max = IMAGE_SIZE * 0.7746;
  b.score = 0.8748;
  bboxes.push_back(b);
  b.x_min = IMAGE_SIZE * 0.0088; b.y_min = IMAGE_SIZE * 0.2738;
  b.x_max = IMAGE_SIZE * 0.9607; b.y_max = IMAGE_SIZE * 0.7555;
  b.score = 0.9192;
  bboxes.push_back(b);
  b.x_min = IMAGE_SIZE * 0.0266; b.y_min = IMAGE_SIZE * 0.2789;
  b.x_max = IMAGE_SIZE * 0.9787; b.y_max = IMAGE_SIZE * 0.7497;
  b.score = 0.8438;
  bboxes.push_back(b);
  b.x_min = IMAGE_SIZE * 0.0379; b.y_min = IMAGE_SIZE * 0.2463;
  b.x_max = IMAGE_SIZE * 0.9559; b.y_max = IMAGE_SIZE * 0.7830;
  b.score = 0.9600;
  bboxes.push_back(b);
  b.x_min = IMAGE_SIZE * 0.0414; b.y_min = IMAGE_SIZE * 0.2487;
  b.x_max = IMAGE_SIZE * 0.9613; b.y_max = IMAGE_SIZE * 0.8001;
  b.score = 0.9560;
  bboxes.push_back(b);
  b.x_min = IMAGE_SIZE * 0.0329; b.y_min = IMAGE_SIZE * 0.2511;
  b.x_max = IMAGE_SIZE * 0.9741; b.y_max = IMAGE_SIZE * 0.7833;
  b.score = 0.9971;
  bboxes.push_back(b);
  b.x_min = IMAGE_SIZE * 0.0387; b.y_min = IMAGE_SIZE * 0.2273;
  b.x_max = IMAGE_SIZE * 0.9698; b.y_max = IMAGE_SIZE * 0.8010;
  b.score = 0.5693;
  bboxes.push_back(b);
  b.x_min = IMAGE_SIZE * 0.0750; b.y_min = IMAGE_SIZE * 0.2530;
  b.x_max = IMAGE_SIZE * 1.0819; b.y_max = IMAGE_SIZE * 0.7849;
  b.score = 0.9879;
  bboxes.push_back(b);
  // clang-format on

  std::vector<int> keeped_indexes = nms(bboxes, .5);
  EXPECT_EQ(keeped_indexes.size(), 1);
  EXPECT_EQ(keeped_indexes[0], 8);
}

}  // namespace tiny_dnn