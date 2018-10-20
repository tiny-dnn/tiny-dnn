/*
  Copyright (c) 2013, Taiga Nomi and the respective contributors
  All rights reserved.

  Use of this source code is governed by a BSD-style license that can be found
  in the LICENSE file.
*/
#pragma once

#include <vector>

namespace tiny_dnn {

struct bounding_box {
  float_t x_min;
  float_t y_min;
  float_t x_max;
  float_t y_max;
  float_t score;

  inline float_t area() const { return (x_max - x_min) * (y_max - y_min); }
};

inline float_t iou(const bounding_box &b1, const bounding_box &b2) {
  float_t x1           = std::max<float_t>(b1.x_min, b2.x_min);
  float_t y1           = std::max<float_t>(b1.y_min, b2.y_min);
  float_t x2           = std::min<float_t>(b1.x_max, b2.x_max);
  float_t y2           = std::min<float_t>(b1.y_max, b2.y_max);
  float_t width        = std::max<float_t>(0.0f, (x2 - x1 + 1));
  float_t height       = std::max<float_t>(0.0f, (y2 - y1 + 1));
  float_t intersection = width * height;
  float_t _iou         = intersection / (b1.area() + b2.area() - intersection);
  return _iou >= 0 ? _iou : 0;
}

inline std::vector<int> nms(std::vector<bounding_box> &proposals,
                     const float_t threshold) {
  std::vector<float_t> scores;
  std::vector<int> indexes;
  for (size_t i = 0; i < proposals.size(); ++i) {
    scores.push_back(proposals[i].score);
    indexes.push_back(i);
  }
  sort(indexes.begin(), indexes.end(),
       [&](int a, int b) { return scores[a] > scores[b]; });

  std::vector<bool> is_keeped(proposals.size(), true);
  for (size_t i = 0; i < proposals.size(); ++i) {
    if (!is_keeped[indexes[i]]) {
      continue;
    }

    for (size_t j = i + 1; j < proposals.size(); ++j) {
      if (iou(proposals[indexes[i]], proposals[indexes[j]]) > threshold) {
        is_keeped[indexes[j]] = false;
      }
    }
  }

  std::vector<int> keeped_bboxes;
  for (int idx : indexes) {
    if (is_keeped[idx]) {
      keeped_bboxes.push_back(idx);
    }
  }
  return keeped_bboxes;
}

}  // namespace tiny_dnn
