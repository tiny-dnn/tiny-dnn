/*
    Copyright (c) 2013, Taiga Nomi
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

TEST(power, forward) {
  power_layer pw(shape3d(3, 2, 1), 2.0, 1.5);

  tensor_t in = {
    {0, 1, 2, 3, 4, 5}, {-5, -4, -3, -2, -1, 0},
  };

  tensor_t out_expected = {{0 * 0 * 1.5, 1 * 1 * 1.5, 2 * 2 * 1.5, 3 * 3 * 1.5,
                            4 * 4 * 1.5, 5 * 5 * 1.5},
                           {5 * 5 * 1.5, 4 * 4 * 1.5, 3 * 3 * 1.5, 2 * 2 * 1.5,
                            1 * 1 * 1.5, 0 * 0 * 1.5}};

  auto out = pw.forward({in});

  for (serial_size_t i = 0; i < 6; i++) {
    EXPECT_FLOAT_EQ(out_expected[0][i], out[0][0][i]);
    EXPECT_FLOAT_EQ(out_expected[1][i], out[0][1][i]);
  }
}

TEST(power, gradient_check) {
  network<sequential> nn;

  nn << fully_connected_layer(10, 20) << tanh()
     << power_layer(shape3d(20, 1, 1), 3.0, 1.5)
     << fully_connected_layer(20, 10) << tanh();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second,
                                     epsilon<float_t>(), GRAD_CHECK_ALL));
}

}  // namespace tiny-dnn
