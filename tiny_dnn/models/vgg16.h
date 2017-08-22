/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <string>

namespace models {

class vgg16 : public tiny_dnn::network<tiny_dnn::sequential> {
 public:
  explicit vgg16(const std::string &name = "vgg16", bool include_top = true)
    : tiny_dnn::network<tiny_dnn::sequential>(name) {
    using conv     = tiny_dnn::layers::conv;
    using relu     = tiny_dnn::activation::relu;
    using max_pool = tiny_dnn::layers::max_pool;
    using dense    = tiny_dnn::layers::dense;

    // Block 1
    *this << conv(224, 224, 3, 3, 3, 64, padding::same) << relu();
    *this << conv(224, 224, 3, 3, 64, 64, padding::same) << relu();
    *this << max_pool(224, 224, 64, 2) << relu();

    // Block 2
    *this << conv(112, 112, 3, 3, 64, 128, padding::same) << relu();
    *this << conv(112, 112, 3, 3, 128, 128, padding::same) << relu();
    *this << max_pool(112, 112, 128, 2) << relu();

    // Block 3
    *this << conv(56, 56, 3, 3, 128, 256, padding::same) << relu();
    *this << conv(56, 56, 3, 3, 256, 256, padding::same) << relu();
    *this << conv(56, 56, 3, 3, 256, 256, padding::same) << relu();
    *this << max_pool(56, 56, 256, 2) << relu();

    // Block 4
    *this << conv(28, 28, 3, 3, 256, 512, padding::same) << relu();
    *this << conv(28, 28, 3, 3, 512, 512, padding::same) << relu();
    *this << conv(28, 28, 3, 3, 512, 512, padding::same) << relu();
    *this << max_pool(28, 28, 512, 2) << relu();

    // Block 5
    *this << conv(14, 14, 3, 3, 512, 512, padding::same) << relu();
    *this << conv(14, 14, 3, 3, 512, 512, padding::same) << relu();
    *this << conv(14, 14, 3, 3, 512, 512, padding::same) << relu();
    *this << max_pool(14, 14, 512, 2) << relu();

    if (include_top) {
      *this << dense(7 * 7 * 512, 4096) << relu();
      *this << dense(4096, 4096) << relu();
      *this << dense(4096, 1000) << relu();
      *this << softmax();
    }
  }
};

}  // namespace models
