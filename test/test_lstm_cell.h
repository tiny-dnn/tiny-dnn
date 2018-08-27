/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <functional>
#include <vector>

namespace tiny_dnn {

TEST(lstm, blackbox_gradient_check) {
  const size_t in_size  = 50;
  const size_t out_size = 10;
  auto l                = recurrent_layer(lstm(in_size, out_size), 1);
  l.reset_state(false);
  std::vector<tensor_t> input_data = generate_test_data(
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {in_size, out_size, out_size, in_size * out_size, in_size * out_size,
     in_size * out_size, in_size * out_size, out_size * out_size,
     out_size * out_size, out_size * out_size, out_size * out_size, out_size,
     out_size, out_size, out_size});
  std::vector<tensor_t> in_grad  = input_data;  // copy constructor
  std::vector<tensor_t> out_data = generate_test_data(
    {1, 1, 1, 1, 1, 1, 1},
    {out_size, out_size, out_size, out_size, out_size, out_size, out_size});
  std::vector<tensor_t> out_grad = generate_test_data(
    {1, 1, 1, 1, 1, 1, 1},
    {out_size, out_size, out_size, out_size, out_size, out_size, out_size});
  const size_t trials = 100;
  for (size_t i = 0; i < trials; i++) {
    const size_t in_edge  = uniform_idx(input_data);
    const size_t in_idx   = uniform_idx(input_data[in_edge][0]);
    const size_t out_edge = uniform_idx(out_data) % 3;
    const size_t out_idx  = uniform_idx(out_data[out_edge][0]);
    float_t ngrad = numeric_gradient(l, input_data, in_edge, in_idx, out_data,
                                     out_edge, out_idx);
    float_t cgrad = analytical_gradient(l, input_data, in_edge, in_idx,
                                        out_data, out_grad, out_edge, out_idx);
    EXPECT_NEAR(ngrad, cgrad, epsilon<float_t>());
  }
}

TEST(lstm, read_write) {
  recurrent_layer l1(lstm(100, 100), 1);
  recurrent_layer l2(lstm(100, 100), 1);

  l1.setup(true);
  l2.setup(true);

  serialization_test(l1, l2);
}
}  // namespace tiny_dnn
