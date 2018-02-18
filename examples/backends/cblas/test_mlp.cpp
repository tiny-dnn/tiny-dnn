/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#include <iostream>
#include <random>

#include "tiny_dnn/tiny_dnn.h"

const int SIZE = 100;

template <typename N>
void construct_net(N &nn, tiny_dnn::core::backend_t backend_type) {
  using relu    = tiny_dnn::relu_layer;
  using fc      = tiny_dnn::fully_connected_layer;
  using softmax = tiny_dnn::softmax_layer;

  nn << fc(SIZE, SIZE, false, backend_type) << relu()
     << fc(SIZE, SIZE, false, backend_type) << softmax();
}

int main(int argc, char **argv) {
  try {
    tiny_dnn::network<tiny_dnn::sequential> nn_internal;
    construct_net(nn_internal, tiny_dnn::core::backend_t::internal);

    tiny_dnn::network<tiny_dnn::sequential> nn_cblas;
    construct_net(nn_cblas, tiny_dnn::core::backend_t::cblas);

    tiny_dnn::vec_t input;
    for (size_t i = 0; i < SIZE; i++) {
      input.push_back(rand_r() / RAND_MAX);
    }
    auto output_internal = nn_internal.fprop(input);
    auto output_cblas    = nn_cblas.fprop(input);

    for (size_t i = 0; i < SIZE; i++) {
      std::cout << output_internal[i] << "|" << output_cblas[i] << "\t";
    }
    std::cout << std::endl;
  } catch (tiny_dnn::nn_error &err) {
    std::cerr << "Exception: " << err.what() << std::endl;
  }
}
