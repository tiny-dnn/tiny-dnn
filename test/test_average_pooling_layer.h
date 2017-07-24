/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once
#include "gtest/gtest.h"
#include "testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn::activation;

namespace tiny_dnn {

TEST(ave_pool, gradient_check) {  // sigmoid - cross-entropy
  using loss_func  = cross_entropy;
  using activation = sigmoid;
  using network    = network<sequential>;

  network nn;
  nn << fully_connected_layer(3, 8) << activation()
     << average_pooling_layer(4, 2, 1, 2) << activation();  // 4x2 => 2x1

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();

  // TODO: gradient check in tests might make it easier to debug
  EXPECT_TRUE(nn.gradient_check<loss_func>(test_data.first, test_data.second,
                                           epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(ave_pool, gradient_check2) {  // x-stride
  using loss_func  = cross_entropy;
  using activation = sigmoid;
  using network    = network<sequential>;

  network nn;
  nn << fully_connected_layer(3, 8) << activation()
     << average_pooling_layer(4, 2, 1, 2, 1, 2, 1)  // 4x2 => 2x2
     << activation();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();

  EXPECT_TRUE(nn.gradient_check<loss_func>(test_data.first, test_data.second,
                                           epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(ave_pool, gradient_check3) {  // y-stride
  using loss_func  = cross_entropy;
  using activation = sigmoid;
  using network    = network<sequential>;

  network nn;
  nn << fully_connected_layer(3, 8) << activation()
     << average_pooling_layer(4, 2, 1, 1, 2, 1, 2)  // 4x2 => 4x1
     << activation();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();

  EXPECT_TRUE(nn.gradient_check<loss_func>(test_data.first, test_data.second,
                                           epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(ave_pool, gradient_check4) {  // padding-same
  using loss_func  = cross_entropy;
  using activation = sigmoid;
  using network    = network<sequential>;

  network nn;
  nn << average_pooling_layer(4, 2, 1, 2, 2, 1, 1, padding::same)
     << activation();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();

  EXPECT_TRUE(nn.gradient_check<loss_func>(test_data.first, test_data.second,
                                           epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(ave_pool, forward) {
  average_pooling_layer l(4, 4, 1, 2);
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

  auto out  = l.forward({{in}});
  vec_t res = (*out[0])[0];

  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_NEAR(expected[i], res[i], 1E-7);
  }
}

TEST(ave_pool, forward_stride) {
  average_pooling_layer l(4, 4, 1, 2, 1);
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

  auto out  = l.forward({{in}});
  vec_t res = (*out[0])[0];

  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_NEAR(expected[i], res[i], 1E-7);
  }
}

TEST(ave_pool, backward) {
  average_pooling_layer l(4, 4, 1, 2);

  l.weight_init(weight_init::constant(1.0));
  l.bias_init(weight_init::constant(1.0));
  l.init_weight();

  // clang-format off
  vec_t in = {
      0, 1, 2, 3,
      8, 7, 5, 6,
      4, 3, 1, 2,
      0,-1,-2,-3
  };

  vec_t curr_delta = {
      1, 2,
      3, 4
  };

  vec_t prev_delta_expected = {
      0.25, 0.25, 0.5, 0.5,
      0.25, 0.25, 0.5, 0.5,
      0.75, 0.75, 1.0, 1.0,
      0.75, 0.75, 1.0, 1.0
  };

  vec_t dw_expected = {
      0, 0,
      0, 0
  };

  vec_t db_expected = {10};
  // clang-format on

  std::vector<tensor_t> in_grad =
    l.backward(std::vector<tensor_t>{{curr_delta}});
  vec_t prev_delta_result = in_grad[0][0];
  vec_t dw_result         = in_grad[1][0];
  vec_t db_result         = in_grad[2][0];

  for (size_t i = 0; i < prev_delta_expected.size(); i++) {
    EXPECT_NEAR(prev_delta_result[i], prev_delta_expected[i], 1E-7);
  }

  for (size_t i = 0; i < dw_expected.size(); i++) {
    EXPECT_NEAR(dw_result[i], dw_expected[i], 1E-7);
  }

  for (size_t i = 0; i < db_expected.size(); i++) {
    EXPECT_NEAR(db_result[i], db_expected[i], 1E-7);
  }
}

TEST(ave_pool, read_write) {
  average_pooling_layer l1(100, 100, 5, 2);
  average_pooling_layer l2(100, 100, 5, 2);

  l1.setup(true);
  l2.setup(true);

  serialization_test(l1, l2);
}

}  // namespace tiny_dnn
