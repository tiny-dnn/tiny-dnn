/*
    Copyright (c) 2017, Taiga Nomi
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <string>
#include <vector>

namespace tiny_dnn {

TEST(global_ave_pool, read_write) {
  global_average_pooling_layer l1(100, 100, 5);
  global_average_pooling_layer l2(100, 100, 5);

  l1.init_weight();
  l2.init_weight();

  serialization_test(l1, l2);
}

TEST(global_ave_pool, forward) {
  global_average_pooling_layer l(4, 4, 1);

  // clang-format off
  vec_t in = {0,  1,  2,  3,
              8,  7,  5,  6,
              4,  3,  1,  2,
              0, -1, -2, -3};
  // clang-format on

  vec_t expected = {2.25};
  std::vector<const tensor_t*> out;
  l.forward({{in}}, out);
  vec_t res = (*out[0])[0];

  EXPECT_FLOAT_EQ(expected[0], res[0]);
}

TEST(global_ave_pool, setup_internal) {
  global_average_pooling_layer l(4, 4, 1, core::backend_t::internal);

  EXPECT_EQ(l.parallelize(), true);          // if layer can be parallelized
  EXPECT_EQ(l.in_channels(), 1u);            // num of input tensors
  EXPECT_EQ(l.out_channels(), 1u);           // num of output tensors
  EXPECT_EQ(l.in_data_size(), 16u);          // size of input tensors
  EXPECT_EQ(l.out_data_size(), 1u);          // size of output tensors
  EXPECT_EQ(l.in_data_shape().size(), 1u);   // num of inputs shapes
  EXPECT_EQ(l.out_data_shape().size(), 1u);  // num of output shapes
  EXPECT_EQ(l.inputs().size(), 1u);          // num of input edges
  EXPECT_EQ(l.outputs().size(), 1u);         // num of output edges
  EXPECT_EQ(l.in_types().size(), 1u);        // num of input data types
  EXPECT_EQ(l.out_types().size(), 1u);       // num of output data types
  EXPECT_EQ(l.fan_in_size(), 16u);           // num of incoming connections
  EXPECT_EQ(l.fan_out_size(), 1u);           // num of outgoing connections
  EXPECT_STREQ(l.layer_type().c_str(),
               "global-ave-pool");  // string with layer type
}

TEST(global_ave_pool, forward_multichannel) {
  global_average_pooling_layer l(2, 2, 3);
  // clang-format off
  vec_t in = {0, 1,
              2, 3, 8, 7,
                    5, 6, 4, 3,
                          1, 2};
  // clang-format on

  vec_t expected = {1.5, 6.5, 2.5};

  std::vector<const tensor_t*> out;
  l.forward({{in}}, out);
  vec_t res = (*out[0])[0];

  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_FLOAT_EQ(expected[i], res[i]);
  }
}

TEST(global_ave_pool, backward) {
  global_average_pooling_layer l(4, 4, 1);
  // clang-format off
  vec_t in = {0,  1,  2,  3,
              8,  7,  5,  6,
              4,  3,  1,  2,
              0, -1, -2, -3};

  vec_t out_grad = { 16 };

  vec_t in_grad_expected = {1, 1, 1, 1,
                            1, 1, 1, 1,
                            1, 1, 1, 1,
                            1, 1, 1, 1};
  // clang-format on
  vec_t in_grad = l.backward(std::vector<tensor_t>{{out_grad}})[0][0];

  for (size_t i = 0; i < in_grad.size(); i++) {
    EXPECT_FLOAT_EQ(in_grad_expected[i], in_grad[i]);
  }
}

TEST(global_ave_pool, backward_multichannel) {
  global_average_pooling_layer l(2, 2, 3);
  // clang-format off
  vec_t in = {0, 1,
              2, 3, 8, 7,
                    5, 6, 4, 3,
                          1, 2};

  vec_t out_grad = {4, 8, 12};

  vec_t in_grad_expected = {1, 1,
                            1, 1, 2, 2,
                                  2, 2, 3, 3,
                                        3, 3};
  // clang-format on

  vec_t in_grad = l.backward(std::vector<tensor_t>{{out_grad}})[0][0];

  for (size_t i = 0; i < in_grad.size(); i++) {
    EXPECT_FLOAT_EQ(in_grad_expected[i], in_grad[i]);
  }
}

TEST(global_ave_pool, gradient_check) {
  const size_t in_width  = 2;
  const size_t in_height = 2;
  const size_t channels  = 3;
  global_average_pooling_layer l(in_width, in_height, channels);
  std::vector<tensor_t> input_data =
    generate_test_data({1}, {in_width * in_height * channels});
  std::vector<tensor_t> in_grad  = input_data;  // copy constructor
  std::vector<tensor_t> out_data = generate_test_data({1}, {channels});
  std::vector<tensor_t> out_grad = generate_test_data({1}, {channels});
  const size_t trials            = 100;
  for (size_t i = 0; i < trials; i++) {
    const size_t in_edge  = uniform_idx(input_data);
    const size_t in_idx   = uniform_idx(input_data[in_edge][0]);
    const size_t out_edge = uniform_idx(out_data);
    const size_t out_idx  = uniform_idx(out_data[out_edge][0]);
    float_t ngrad = numeric_gradient(l, input_data, in_edge, in_idx, out_data,
                                     out_edge, out_idx);
    float_t cgrad = analytical_gradient(l, input_data, in_edge, in_idx,
                                        out_data, out_grad, out_edge, out_idx);
    EXPECT_NEAR(ngrad, cgrad, epsilon<float_t>());
  }
}

}  // namespace tiny_dnn
