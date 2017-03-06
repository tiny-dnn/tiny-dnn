/*
    Copyright (c) 2016, Taiga Nomi
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once
#include "gtest/gtest.h"
#include "testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

namespace tiny_dnn {

TEST(ave_pool, gradient_check) {  // sigmoid - cross-entropy
  typedef cross_entropy loss_func;
  typedef activation::sigmoid activation;
  typedef network<sequential> network;

  network nn;
  nn << fully_connected_layer<activation>(3, 8)
     << average_pooling_layer<activation>(4, 2, 1, 2);  // 4x2 => 2x1

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();

  EXPECT_TRUE(nn.gradient_check<loss_func>(test_data.first, test_data.second,
                                           epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(ave_pool, gradient_check2) {  // x-stride
  typedef cross_entropy loss_func;
  typedef activation::sigmoid activation;
  typedef network<sequential> network;

  network nn;
  nn << fully_connected_layer<activation>(3, 8)
     << average_pooling_layer<activation>(4, 2, 1, 2, 1, 2, 1);  // 4x2 => 2x2

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();

  EXPECT_TRUE(nn.gradient_check<loss_func>(test_data.first, test_data.second,
                                           epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(ave_pool, gradient_check3) {  // y-stride
  typedef cross_entropy loss_func;
  typedef activation::sigmoid activation;
  typedef network<sequential> network;

  network nn;
  nn << fully_connected_layer<activation>(3, 8)
     << average_pooling_layer<activation>(4, 2, 1, 1, 2, 1, 2);  // 4x2 => 4x1

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();

  EXPECT_TRUE(nn.gradient_check<loss_func>(test_data.first, test_data.second,
                                           epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(ave_pool, gradient_check4) {  // padding-same
  typedef cross_entropy loss_func;
  typedef activation::sigmoid activation;
  typedef network<sequential> network;

  network nn;
  nn << average_pooling_layer<activation>(4, 2, 1, 2, 2, 1, 1,
                                          padding::same);  // 4x2 => 4x1

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();

  EXPECT_TRUE(nn.gradient_check<loss_func>(test_data.first, test_data.second,
                                           epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(ave_pool, forward) {
  average_pooling_layer<identity> l(4, 4, 1, 2);
  // clang-format off
    vec_t in = {
        0, 1, 2, 3,
        8, 7, 5, 6,
        4, 3, 1, 2,
        0,-1,-2,-3
    };

    vec_t expected = {
        4, 4,
        1.5, -0.5
    };
  // clang-format on

  l.weight_init(weight_init::constant(1.0));
  l.bias_init(weight_init::constant(0.0));
  l.init_weight();

  vec_t res = l.forward({{in}})[0][0];

  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_FLOAT_EQ(expected[i], res[i]);
  }
}

TEST(ave_pool, forward_stride) {
  average_pooling_layer<identity> l(4, 4, 1, 2, 1);
  // clang-format off
    vec_t in = {
        0, 1, 2, 3,
        8, 7, 5, 6,
        4, 3, 1, 2,
        0,-1,-2,-3
    };

    vec_t expected = {
        16.0/4, 15.0/4, 16.0/4,
        22.0/4, 16.0/4, 14.0/4,
         6.0/4,  1.0/4, -2.0/4
    };
  // clang-format on

  l.weight_init(weight_init::constant(1.0));
  l.bias_init(weight_init::constant(0.0));
  l.init_weight();

  vec_t res = l.forward({{in}})[0][0];

  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_FLOAT_EQ(expected[i], res[i]);
  }
}

TEST(ave_pool, read_write) {
  average_pooling_layer<tan_h> l1(100, 100, 5, 2);
  average_pooling_layer<tan_h> l2(100, 100, 5, 2);

  l1.setup(true);
  l2.setup(true);

  serialization_test(l1, l2);
}

}  // namespace tiny-dnn
