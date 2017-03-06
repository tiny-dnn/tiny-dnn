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

namespace tiny_dnn {

TEST(concat, forward_data) {
  std::vector<shape3d> in_shapes;
  in_shapes.push_back(shape3d(1, 2, 1));
  in_shapes.push_back(shape3d(1, 2, 1));
  in_shapes.push_back(shape3d(1, 2, 1));
  concat_layer cl(in_shapes);

  // clang-format off
    tensor_t in0 = {
        { 0, 1 },
        { 6, 7 },
        { 12, 13 },
        { 18, 19 }
    };

    tensor_t in1 = {
        {  2, 3 },
        {  8, 9 },
        { 14, 15 },
        { 20, 21 }
    };

    tensor_t in2 = {
        { 4, 5 },
        { 10, 11 },
        { 16, 17 },
        { 22, 23 }
    };
    
    tensor_t out_expected = {
        { 0, 1, 2, 3, 4, 5 },
        { 6, 7, 8, 9, 10, 11 },
        { 12, 13, 14, 15, 16, 17 },
        { 18, 19, 20, 21, 22, 23 }
    };

  // clang-format on

  auto out = cl.forward({in0, in1, in2});

  for (serial_size_t i = 0; i < 4; i++) {
    for (serial_size_t j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(out_expected[i][j], out[0][i][j]);
    }
  }

  out = cl.backward({out_expected});

  for (serial_size_t i = 0; i < 4; i++) {
    for (serial_size_t j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(in0[i][j], out[0][i][j]);
      EXPECT_FLOAT_EQ(in1[i][j], out[1][i][j]);
      EXPECT_FLOAT_EQ(in2[i][j], out[2][i][j]);
    }
  }
}

}  // namespace tiny-dnn
