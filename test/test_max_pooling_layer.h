// Copyright (c) 2013-2016, Taiga Nomi. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#pragma once

#include <string>

#include "picotest/picotest.h"
#include "testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

namespace tiny_dnn {

TEST(max_pool, read_write) {
  max_pooling_layer<tan_h> l1(100, 100, 5, 2);
  max_pooling_layer<tan_h> l2(100, 100, 5, 2);

  l1.init_weight();
  l2.init_weight();

  serialization_test(l1, l2);
}

TEST(max_pool, forward) {
  max_pooling_layer<identity> l(4, 4, 1, 2);
  vec_t in = {0, 1, 2, 3, 8, 7, 5, 6, 4, 3, 1, 2, 0, -1, -2, -3};

  vec_t expected = {8, 6, 4, 2};

  vec_t res = l.forward({{in}})[0][0];

  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_FLOAT_EQ(expected[i], res[i]);
  }
}

TEST(max_pool, setup_tiny) {
  max_pooling_layer<identity> l(4, 4, 1, 2, 2, core::backend_t::tiny_dnn);

  EXPECT_EQ(l.parallelize(), true);             // if layer can be parallelized
  EXPECT_EQ(l.in_channels(), cnn_size_t(1));    // num of input tensors
  EXPECT_EQ(l.out_channels(), cnn_size_t(2));   // num of output tensors
  EXPECT_EQ(l.in_data_size(), cnn_size_t(16));  // size of input tensors
  EXPECT_EQ(l.out_data_size(), cnn_size_t(4));  // size of output tensors
  EXPECT_EQ(l.in_data_shape().size(), cnn_size_t(1));   // num of inputs shapes
  EXPECT_EQ(l.out_data_shape().size(), cnn_size_t(1));  // num of output shapes
  EXPECT_EQ(l.weights().size(), cnn_size_t(0));  // the wieghts vector size
  EXPECT_EQ(l.weights_grads().size(),
            cnn_size_t(0));                        // the wieghts vector size
  EXPECT_EQ(l.inputs().size(), cnn_size_t(1));     // num of input edges
  EXPECT_EQ(l.outputs().size(), cnn_size_t(2));    // num of outpus edges
  EXPECT_EQ(l.in_types().size(), cnn_size_t(1));   // num of input data types
  EXPECT_EQ(l.out_types().size(), cnn_size_t(2));  // num of output data types
  EXPECT_EQ(l.fan_in_size(), cnn_size_t(4));   // num of incoming connections
  EXPECT_EQ(l.fan_out_size(), cnn_size_t(1));  // num of outgoing connections
  EXPECT_STREQ(l.layer_type().c_str(), "max-pool");  // string with layer type
}

TEST(max_pool, forward_stride_tiny) {
  max_pooling_layer<identity> l(4, 4, 1, 2, 2, core::backend_t::tiny_dnn);
  vec_t in = {0, 1, 2, 3, 8, 7, 5, 6, 4, 3, 1, 2, 0, -1, -2, -3};

  vec_t expected = {8, 6, 4, 2};

  vec_t res = l.forward({{in}})[0][0];

  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_FLOAT_EQ(expected[i], res[i]);
  }
}

#ifdef CNN_USE_NNPACK
TEST(max_pool, forward_stride_nnp) {
  max_pooling_layer<identity> l(4, 4, 1, 2, 2, core::backend_t::nnpack);
  vec_t in = {0, 1, 2, 3, 8, 7, 5, 6, 4, 3, 1, 2, 0, -1, -2, -3};

  vec_t expected = {8, 6, 4, 2};

  vec_t res = l.forward({{in}})[0][0];

  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_FLOAT_EQ(expected[i], res[i]);
  }
}
#endif

TEST(max_pool, forward_stride) {
  max_pooling_layer<identity> l(4, 4, 1, 2, 1);
  vec_t in = {0, 1, 2, 3, 8, 7, 5, 6, 4, 3, 1, 2, 0, -1, -2, -3};

  vec_t expected = {8, 7, 6, 8, 7, 6, 4, 3, 2};

  vec_t res = l.forward({{in}})[0][0];

  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_FLOAT_EQ(expected[i], res[i]);
  }
}

TEST(max_pool, backward) {
  max_pooling_layer<identity> l(4, 4, 1, 2);
  vec_t in = {0, 1, 2, 3, 8, 7, 5, 6, 4, 3, 1, 2, 0, -1, -2, -3};

  vec_t out_grad = {1, 2, 3, 4};

  vec_t in_grad_expected = {0, 0, 0, 0, 1, 0, 0, 2, 3, 0, 0, 4, 0, 0, 0, 0};

  l.forward({{in}})[0];
  vec_t in_grad = l.backward(std::vector<tensor_t>{{out_grad}})[0][0];

  for (size_t i = 0; i < in_grad.size(); i++) {
    EXPECT_FLOAT_EQ(in_grad_expected[i], in_grad[i]);
  }
}

}  // namespace tiny_dnn
