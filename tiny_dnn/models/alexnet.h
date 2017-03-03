/*
    Copyright (c) 2013, Taiga Nomi
    Copyright (c) 2016, Taiga Nomi, Edgar Riba
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/

// #include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::layers;

namespace models {

// Based on:
// https://github.com/DeepMark/deepmark/blob/master/torch/image%2Bvideo/alexnet.lua
class alexnet : public network<sequential> {
 public:
  explicit alexnet(const std::string &name = "") : network<sequential>(name) {
    // todo: (karandesai) shift this to tiny_dnn::activation
    using relu = relu_layer;
    *this << conv(224, 224, 11, 11, 3, 64, padding::valid, true, 4, 4);
    *this << relu(54, 54, 64);
    *this << max_pool(54, 54, 64, 2);
    *this << conv(27, 27, 5, 5, 64, 192, padding::valid, true, 1, 1);
    *this << relu(23, 23, 192);
    *this << max_pool(23, 23, 192, 1);
    *this << conv(23, 23, 3, 3, 192, 384, padding::valid, true, 1, 1);
    *this << relu(21, 21, 384);
    *this << conv(21, 21, 3, 3, 384, 256, padding::valid, true, 1, 1);
    *this << relu(19, 19, 256);
    *this << conv(19, 19, 3, 3, 256, 256, padding::valid, true, 1, 1);
    *this << relu(17, 17, 256);
    *this << max_pool(17, 17, 256, 1);
  }
};

}  // namespace models
