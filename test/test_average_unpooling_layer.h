/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <gtest/gtest.h>
#include <vector>

#include "test/testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

namespace tiny_dnn {

TEST(ave_unpool, forward) {
  average_unpooling_layer l(2, 2, 1, 2);
  l.weight_init_f(parameter_init::constant(1.0));
  l.bias_init_f(parameter_init::constant(0.0));
  l.init_parameters();

  // clang-format off
    vec_t in = {
        4, 3,
        1.5, -0.5
    };

    vec_t expected = {
        4,     4,    3,    3,
        4,     4,    3,    3,
        1.5, 1.5, -0.5, -0.5,
        1.5, 1.5, -0.5, -0.5,
    };
  // clang-format on

  auto out  = l.forward({{in}});
  vec_t res = (*out[0])[0];

  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_FLOAT_EQ(expected[i], res[i]);
  }
}

TEST(ave_unpool, forward_stride) {
  average_unpooling_layer l(3, 3, 1, 2, 1);
  l.weight_init_f(parameter_init::constant(1.0));
  l.bias_init_f(parameter_init::constant(0.0));
  l.init_parameters();

  // clang-format off
    vec_t in = {
        0, 1, 2,
        8, 7, 5,
        4, 3, 1,
    };

    vec_t expected = {
        0,   1,  3, 2,
        8,  16, 15, 7,
        12, 22, 16, 6,
        4,  7,  4,  1
    };
  // clang-format on

  auto out  = l.forward({{in}});
  vec_t res = (*out[0])[0];

  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_FLOAT_EQ(expected[i], res[i]);
  }
}

TEST(ave_unpool, gradient_check) {  // sigmoid - mse
  using loss_func  = mse;           // TODO(Randl): fails with cross-entropy
  using activation = sigmoid;
  using network    = network<sequential>;

  network nn;
  nn << fully_connected_layer(3, 4) << activation()
     << average_unpooling_layer(2, 2, 1, 2)  // 2x2 => 4x4
     << activation() << average_pooling_layer(4, 4, 1, 2) << activation();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();

  EXPECT_TRUE(nn.gradient_check<loss_func>(test_data.first, test_data.second,
                                           epsilon<float_t>(), GRAD_CHECK_ALL));
}

/* todo (karandesai) Deal with serialization tests of all layers together later.
TEST(ave_unpool, read_write) {
  average_unpooling_layer l1(100, 100, 5, 2);
  average_unpooling_layer l2(100, 100, 5, 2);

  l1.init_parameters();
  l2.init_parameters();

  serialization_test(l1, l2);
}
*/
}  // namespace tiny_dnn
