/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <vector>

namespace tiny_dnn {

TEST(ave_pool, forward) {
  average_pooling_layer l(4, 4, 1, 2);
  // clang-format off
    vec_t in = {
        0,  1,  2,  3,
        8,  7,  5,  6,
        4,  3,  1,  2,
        0, -1, -2, -3
    };

    vec_t expected = {
        4, 4,
        1.5, -0.5
    };
  // clang-format on

  l.weight_init(weight_init::constant(1.0));
  l.bias_init(weight_init::constant(0.0));
  l.init_weight();

  std::vector<const tensor_t*> out;
  l.forward({{in}}, out);
  vec_t res = (*out[0])[0];

  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_FLOAT_EQ(expected[i], res[i]);
  }
}

TEST(ave_pool, forward_stride) {
  average_pooling_layer l(4, 4, 1, 2, 1, false);
  // clang-format off
    vec_t in = {
        0,  1,  2,  3,
        8,  7,  5,  6,
        4,  3,  1,  2,
        0, -1, -2, -3
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

  std::vector<const tensor_t*> out;
  l.forward({{in}}, out);
  vec_t res = (*out[0])[0];

  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_FLOAT_EQ(expected[i], res[i]);
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
