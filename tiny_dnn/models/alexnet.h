/*
    Copyright (c) 2013, Taiga Nomi
    Copyright (c) 2016, Taiga Nomi, Edgar Riba
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/

// #include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

namespace models {

// Based on:
// https://github.com/DeepMark/deepmark/blob/master/torch/image%2Bvideo/alexnet.lua
class alexnet : public network<sequential> {
 public:
  explicit alexnet(const std::string &name = "") : network<sequential>(name) {
    *this << conv<relu>(224, 224, 11, 11, 3, 64, padding::valid, true, 4, 4);
    *this << max_pool<identity>(54, 54, 64, 2);
    *this << conv<relu>(27, 27, 5, 5, 64, 192, padding::valid, true, 1, 1);
    *this << max_pool<identity>(23, 23, 192, 1);
    *this << conv<relu>(23, 23, 3, 3, 192, 384, padding::valid, true, 1, 1);
    *this << conv<relu>(21, 21, 3, 3, 384, 256, padding::valid, true, 1, 1);
    *this << conv<relu>(19, 19, 3, 3, 256, 256, padding::valid, true, 1, 1);
    *this << max_pool<identity>(17, 17, 256, 1);
  }
};

}  // namespace models
