/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <vector>

namespace tiny_dnn {

TEST(selu, gradient_check) {
  const size_t width    = 3;
  const size_t height   = 3;
  const size_t channels = 10;
  selu slu(width, height, channels);
  std::vector<tensor_t> input_data =
    generate_test_data({1}, {width * height * channels});
  std::vector<tensor_t> in_grad = input_data;  // copy constructor
  std::vector<tensor_t> out_data =
    generate_test_data({1}, {width * height * channels});
  std::vector<tensor_t> out_grad =
    generate_test_data({1}, {width * height * channels});
  const size_t trials = 100;
  for (size_t i = 0; i < trials; i++) {
    const size_t in_edge  = uniform_idx(input_data);
    const size_t in_idx   = uniform_idx(input_data[in_edge][0]);
    const size_t out_edge = uniform_idx(out_data);
    const size_t out_idx  = uniform_idx(out_data[out_edge][0]);
    float_t ngrad = numeric_gradient(slu, input_data, in_edge, in_idx, out_data,
                                     out_edge, out_idx);
    float_t cgrad = analytical_gradient(slu, input_data, in_edge, in_idx,
                                        out_data, out_grad, out_edge, out_idx);
    EXPECT_NEAR(ngrad, cgrad, epsilon<float_t>());
  }
}

}  // namespace tiny_dnn
