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
  float x_min;
  float y_min;
  float x_max;
  float y_max;

  inline float area() const { return (x_max - x_min) * (y_max - y_min); }
};

float iou(const bounding_box &b1, const bounding_box &b2) {
  float x1           = std::max(b1.x_min, b2.x_min);
  float y1           = std::max(b1.y_min, b2.y_min);
  float x2           = std::min(b1.x_max, b2.x_max);
  float y2           = std::min(b1.y_max, b2.y_max);
  float width        = std::max(0.0f, (x2 - x1 + 1));
  float height       = std::max(0.0f, (y2 - y1 + 1));
  float intersection = width * height;
  float _iou         = intersection / (b1.area() + b2.area() - intersection);
  return _iou >= 0 ? _iou : 0;
}

std::vector<int> nms(std::vector<bounding_box> &proposals,
                     const float threshold) {
  std::vector<float> areas;
  std::vector<int> indexes;
  for (size_t i = 0; i < proposals.size(); ++i) {
    areas.push_back(proposals[i].area());
    indexes.push_back(i);
  }
  sort(indexes.begin(), indexes.end(),
       [&](int a, int b) { return areas[a] > areas[b]; });

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
