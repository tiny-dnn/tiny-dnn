/*
    Copyright (c) 2016, Taiga Nomi, Edgar Riba
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once
#include "gtest/gtest.h"
#include "testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;

namespace tiny_dnn {

TEST(models, alexnet) {
  models::alexnet nn("alexnet");

  ASSERT_EQ(nn.name(), "alexnet");
  EXPECT_EQ(nn.in_data_size(), serial_size_t(224 * 224 * 3));

  vec_t in(nn.in_data_size());

  // generate random variables
  uniform_rand(in.begin(), in.end(), 0, 1);

  // init wieghts and biases
  nn.weight_init(weight_init::constant(2.0));
  nn.bias_init(weight_init::constant(2.0));
  nn.init_weight();

  // predict
  auto res = nn.predict(in);
}

}  // namespace tiny-dnn
