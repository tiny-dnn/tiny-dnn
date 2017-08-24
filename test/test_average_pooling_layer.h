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

TEST(ave_pool, gradient_check) {
  const size_t in_width    = 16;
  const size_t in_height   = 16;
  const size_t kernel_size = 2;
  const size_t in_channels = 4;

  average_pooling_layer avepool(in_width, in_height, in_channels, kernel_size);

  avepool.init_parameters();

  std::vector<tensor_t> input_data =
    generate_test_data({1}, {in_width * in_height * in_channels});
  std::vector<tensor_t> in_grad  = input_data;  // copy constructor
  std::vector<tensor_t> out_data = generate_test_data(
    {1}, {in_width / kernel_size * in_height / kernel_size * in_channels});
  std::vector<tensor_t> out_grad = generate_test_data(
    {1}, {in_width / kernel_size * in_height / kernel_size * in_channels});

  const size_t trials = 100;
  for (size_t i = 0; i < trials; i++) {
    const size_t in_edge  = uniform_idx(input_data);
    const size_t in_idx   = uniform_idx(input_data[in_edge][0]);
    const size_t out_edge = uniform_idx(out_data);
    const size_t out_idx  = uniform_idx(out_data[out_edge][0]);
    float_t ngrad = numeric_gradient(avepool, input_data, in_edge, in_idx,
                                     out_data, out_grad, out_edge, out_idx);
    float_t cgrad = analytical_gradient(avepool, input_data, in_edge, in_idx,
                                        out_data, out_grad, out_edge, out_idx);
    EXPECT_NEAR(ngrad, cgrad, epsilon<float_t>());
  }
}

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

  l.weight_init(parameter_init::constant(1.0));
  l.bias_init(parameter_init::constant(0.0));
  l.init_parameters();

  auto out  = l.forward({{Tensor<>(tensor_t{{in}})}});
  vec_t res = out[0]->toVec();

  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_NEAR(expected[i], res[i], 1E-7);
  }
}

TEST(ave_pool, forward_stride) {
  average_pooling_layer l(4, 4, 1, 2, 1);
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

  l.weight_init(parameter_init::constant(1.0));
  l.bias_init(parameter_init::constant(0.0));
  l.init_parameters();

  auto out  = l.forward({{Tensor<>(tensor_t{{in}})}});
  vec_t res = (*out[0]).toTensor()[0];

  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_NEAR(expected[i], res[i], 1E-7);
  }
}

TEST(ave_pool, backward) {
  average_pooling_layer l(4, 4, 1, 2);

  l.weight_init(parameter_init::constant(1.0));
  l.bias_init(parameter_init::constant(1.0));
  l.setup(false);

  // clang-format off
  vec_t in = {
      0,  1,  2,  3,
      8,  7,  5,  6,
      4,  3,  1,  2,
      0, -1, -2, -3
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

  auto in_grad =
    l.backward(std::vector<Tensor<>>{{Tensor<>(tensor_t{{curr_delta}})}});
  vec_t prev_delta_result = in_grad[0].toTensor()[0];
  vec_t dw_result         = l.weights_at()[0]->data()->toTensor()[0];
  vec_t db_result         = l.bias_at()[0]->data()->toTensor()[0];

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

  l1.init_parameters();
  l2.init_parameters();

  serialization_test(l1, l2);
}

}  // namespace tiny_dnn
