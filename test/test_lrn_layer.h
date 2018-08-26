/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <vector>

namespace tiny_dnn {

TEST(lrn, cross) {
  lrn_layer lrn(1, 1, 3, 4, /*alpha=*/1.5, /*beta=*/2.0,
                norm_region::across_channels);

  tiny_dnn::float_t in[4]       = {-1.0, 3.0, 2.0, 5.0};
  tiny_dnn::float_t expected[4] = {
    -1.0 / 36.0,       // -1.0 / (1+0.5*(1*1+3*3))^2
    3.0 / 64.0,        //  3.0 / (1+0.5*(1*1+3*3+2*2))^2
    2.0 / 400.0,       //  2.0 / (1+0.5*(3*3+2*2+5*5))^2
    5.0 / 15.5 / 15.5  // 5.0 / (1+0.5*(2*2+5*5))^2
  };
  std::vector<const tensor_t*> o;
  lrn.forward({{vec_t(in, in + 4)}}, o);
  auto out = (*o[0])[0];

  EXPECT_NEAR(expected[0], out[0], epsilon<float_t>());
  EXPECT_NEAR(expected[1], out[1], epsilon<float_t>());
  EXPECT_NEAR(expected[2], out[2], epsilon<float_t>());
  EXPECT_NEAR(expected[3], out[3], epsilon<float_t>());
}

/* Backprop not implemented
TEST(lrn, gradient_check) {
  const size_t in_width = 2;
  const size_t in_height = 2;
  const size_t local_size = 3;
  const size_t channels = 4;
  lrn_layer lrn(in_width, in_height, local_size, channels);
  std::vector<tensor_t> input_data = generate_test_data({1},
{in_width*in_height*channels});
  std::vector<tensor_t> in_grad = input_data;  // copy constructor
  std::vector<tensor_t> out_data = generate_test_data({1},
{in_width*in_height*channels});
  std::vector<tensor_t> out_grad = generate_test_data({1},
{in_width*in_height*channels});
  const size_t trials = 100;
  for (size_t i = 0; i < trials; i++) {
    const size_t in_edge = uniform_idx(input_data);
    const size_t in_idx = uniform_idx(input_data[in_edge][0]);
    const size_t out_edge = uniform_idx(out_data);
    const size_t out_idx = uniform_idx(out_data[out_edge][0]);
    float_t ngrad = numeric_gradient(lrn, input_data, in_edge, in_idx, out_data,
out_grad, out_edge, out_idx);
    float_t cgrad = analytical_gradient(lrn, input_data, in_edge, in_idx,
out_data, out_grad, out_edge, out_idx);
    EXPECT_NEAR(ngrad, cgrad, epsilon<float_t>());
  }
}
*/

TEST(lrn, read_write) {
  lrn_layer l1(10, 10, 3, 4, 1.5, 2.0, norm_region::across_channels);
  lrn_layer l2(10, 10, 3, 4, 1.5, 2.0, norm_region::across_channels);

  l1.init_weight();
  l2.init_weight();

  serialization_test(l1, l2);
}

}  // namespace tiny_dnn
