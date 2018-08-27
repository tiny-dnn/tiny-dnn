/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <vector>

namespace tiny_dnn {

TEST(zero_pad, forward) {
  zero_pad_layer zpl(shape3d(5, 5, 2), 1, 2);

  // clang-format off
  vec_t in = {
    // Channel 0
      0, 1, 2, 3, 4, 
      5, 6, 7, 8, 9,
      10, 11, 12, 13, 14,
      15, 16, 17, 18, 19,
      20, 21, 22, 23, 24,
    // Channel 1
      25, 26, 27, 28, 29,
      30, 31, 32, 33, 34,
      35, 36, 37, 38, 39,
      40, 41, 42, 43, 44,
      45, 46, 47, 48, 49,
  };

  vec_t out_expected = {
    // Channel 0
      0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 2, 3, 4, 0,
      0, 5, 6, 7, 8, 9, 0,
      0, 10, 11, 12, 13, 14, 0,
      0, 15, 16, 17, 18, 19, 0,
      0, 20, 21, 22, 23, 24, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0,
    // Channel 1
      0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 25, 26, 27, 28, 29, 0,
      0, 30, 31, 32, 33, 34, 0,
      0, 35, 36, 37, 38, 39, 0,
      0, 40, 41, 42, 43, 44, 0,
      0, 45, 46, 47, 48, 49, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0
  };
  // clang-format on

  EXPECT_EQ(zpl.in_shape()[0], shape3d(5, 5, 2));
  EXPECT_EQ(zpl.out_shape()[0], shape3d(7, 9, 2));

  {
    std::vector<const tensor_t*> out;
    zpl.forward({{in}}, out);

    vec_t res = (*out[0])[0];
    EXPECT_EQ(out_expected.size(), res.size());

    for (size_t i = 0; i < out_expected.size(); ++i) {
      EXPECT_EQ(out_expected[i], res[i]);
    }
  }

  {
    std::vector<tensor_t> out = zpl.backward({{out_expected}});

    vec_t res = out[0][0];
    EXPECT_EQ(in.size(), res.size());

    for (size_t i = 0; i < in.size(); ++i) {
      EXPECT_EQ(in[i], res[i]);
    }
  }
}

}  // namespace tiny_dnn
