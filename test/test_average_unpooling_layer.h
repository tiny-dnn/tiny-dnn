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

TEST(ave_unpool, gradient_check) {
  const size_t in_width    = 8;
  const size_t in_height   = 8;
  const size_t kernel_size = 2;
  const size_t in_channels = 4;

  average_unpooling_layer aveunpool(in_width, in_height, in_channels,
                                    kernel_size);

  aveunpool.init_parameters();

  std::vector<tensor_t> input_data =
    generate_test_data({1}, {in_width * in_height * in_channels});
  std::vector<tensor_t> in_grad  = input_data;  // copy constructor
  std::vector<tensor_t> out_data = generate_test_data(
    {1}, {in_width * kernel_size * in_height * kernel_size * in_channels});
  std::vector<tensor_t> out_grad = generate_test_data(
    {1}, {in_width * kernel_size * in_height * kernel_size * in_channels});

  const size_t trials = 100;
  for (size_t i = 0; i < trials; i++) {
    const size_t in_edge  = uniform_idx(input_data);
    const size_t in_idx   = uniform_idx(input_data[in_edge][0]);
    const size_t out_edge = uniform_idx(out_data);
    const size_t out_idx  = uniform_idx(out_data[out_edge][0]);
    float_t ngrad = numeric_gradient(aveunpool, input_data, in_edge, in_idx,
                                     out_data, out_grad, out_edge, out_idx);
    float_t cgrad = analytical_gradient(aveunpool, input_data, in_edge, in_idx,
                                        out_data, out_grad, out_edge, out_idx);
    EXPECT_NEAR(ngrad, cgrad, epsilon<float_t>());
  }
}

TEST(ave_unpool, forward) {
  average_unpooling_layer l(2, 2, 1, 2);
  l.weight_init(parameter_init::constant(1.0));
  l.bias_init(parameter_init::constant(0.0));
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

  std::vector<const Tensor<>*> out;
  l.forward({{Tensor<>(tensor_t{{in}})}}, out);
  vec_t res = out[0]->toVec();
  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_FLOAT_EQ(expected[i], res[i]);
  }
}

TEST(ave_unpool, forward_stride) {
  average_unpooling_layer l(3, 3, 1, 2, 1);
  l.weight_init(parameter_init::constant(1.0));
  l.bias_init(parameter_init::constant(0.0));
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

  std::vector<const Tensor<>*> out;
  l.forward({{Tensor<>(tensor_t{{in}})}}, out);
  vec_t res = (*out[0]).toTensor()[0];

  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_FLOAT_EQ(expected[i], res[i]);
  }
}

TEST(ave_unpool, read_write) {
  average_unpooling_layer l1(100, 100, 5, 2);
  average_unpooling_layer l2(100, 100, 5, 2);

  l1.init_parameters();
  l2.init_parameters();

  serialization_test(l1, l2);
}
}  // namespace tiny_dnn
