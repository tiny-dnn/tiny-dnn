/*
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <gtest/gtest.h>

#include <functional>
#include <vector>

#include "test/testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

namespace tiny_dnn {

TEST(recurrent_cell, gradient_check) {
  const size_t in_size  = 50;
  const size_t out_size = 10;
  recurrent_cell_layer l(in_size, out_size);
  l.weight_init(parameter_init::gaussian());
  l.bias_init(parameter_init::gaussian());
  l.init_parameters();

  std::vector<tensor_t> input_data =
    generate_test_data({1, 1}, {in_size, out_size});
  std::vector<tensor_t> in_grad = input_data;  // copy constructor
  std::vector<tensor_t> out_data =
    generate_test_data({1, 1}, {out_size, out_size});
  std::vector<tensor_t> out_grad =
    generate_test_data({1, 1}, {out_size, out_size});
  const size_t trials = 100;
  for (size_t i = 0; i < trials; i++) {
    const size_t in_edge  = uniform_idx(input_data);
    const size_t in_idx   = uniform_idx(input_data[in_edge][0]);
    const size_t out_edge = uniform_idx(out_data);
    const size_t out_idx  = uniform_idx(out_data[out_edge][0]);
    float_t ngrad = numeric_gradient(l, input_data, in_edge, in_idx, out_data,
                                     out_grad, out_edge, out_idx);
    float_t cgrad = analytical_gradient(l, input_data, in_edge, in_idx,
                                        out_data, out_grad, out_edge, out_idx);
    EXPECT_NEAR(ngrad, cgrad, epsilon<float_t>());
  }
}

TEST(recurrent_cell, train) {
  network<sequential> nn;
  adagrad optimizer;

  nn << recurrent_cell_layer(3, 2, true, new tanh_layer) << sigmoid();
  nn.init_parameters();
  vec_t a(3), t(2), a2(3), t2(2);

  // clang-format off
    a[0] = 3.0; a[1] = 0.0; a[2] = -1.0;
    t[0] = 0.3; t[1] = 0.7;

    a2[0] = 0.2; a2[1] = 0.5; a2[2] = 4.0;
    t2[0] = 0.5; t2[1] = 0.1;
  // clang-format on

  std::vector<vec_t> data, train;

  for (size_t i = 0; i < 100; i++) {
    data.push_back(a);
    data.push_back(a2);
    train.push_back(t);
    train.push_back(t2);
  }
  optimizer.alpha = 0.1;
  nn.train<mse>(optimizer, data, train, 1, 20);

  vec_t predicted = nn.predict(a);

  EXPECT_NEAR(predicted[0], t[0], 1e-5);
  EXPECT_NEAR(predicted[1], t[1], 1e-5);

  predicted = nn.predict(a2);

  EXPECT_NEAR(predicted[0], t2[0], 1e-5);
  EXPECT_NEAR(predicted[1], t2[1], 1e-5);
}

TEST(recurrent_cell, train_different_batches) {
  auto batch_sizes = {2, 7, 10, 12};
  size_t data_size = std::accumulate(batch_sizes.begin(), batch_sizes.end(), 1,
                                     std::multiplies<size_t>());
  for (auto &batch_sz : batch_sizes) {
    network<sequential> nn;
    adagrad optimizer;

    nn << recurrent_cell_layer(3, 2) << sigmoid();
    nn.init_parameters();
    vec_t a(3), t(2), a2(3), t2(2);

    // clang-format off
    a[0] = 3.0; a[1] = 0.0; a[2] = -1.0;
    t[0] = 0.3; t[1] = 0.7;

    a2[0] = 0.2; a2[1] = 0.5; a2[2] = 4.0;
    t2[0] = 0.5; t2[1] = 0.1;
    // clang-format on

    std::vector<vec_t> data, train;

    for (size_t i = 0; i < data_size; i++) {
      data.push_back(a);
      data.push_back(a2);
      train.push_back(t);
      train.push_back(t2);
    }
    optimizer.alpha = 0.1;
    nn.train<mse>(optimizer, data, train, batch_sz, 10);

    vec_t predicted = nn.predict(a);

    EXPECT_NEAR(predicted[0], t[0], 1e-5);
    EXPECT_NEAR(predicted[1], t[1], 1e-5);

    predicted = nn.predict(a2);

    EXPECT_NEAR(predicted[0], t2[0], 1e-5);
    EXPECT_NEAR(predicted[1], t2[1], 1e-5);
  }
}

TEST(recurrent_cell, train2) {
  network<sequential> nn;
  gradient_descent optimizer;

  nn << recurrent_cell_layer(4, 6) << tanh_layer() << recurrent_cell_layer(6, 3)
     << tanh_layer();
  nn.init_parameters();
  vec_t a(4, 0.0), t(3, 0.0), a2(4, 0.0), t2(3, 0.0);

  // clang-format off
    a[0] = 3.0; a[1] = 1.0; a[2] = -1.0; a[3] = 4.0;
    t[0] = 0.3; t[1] = 0.7; t[2] = 0.3;

    a2[0] = 1.0; a2[1] = 0.0; a2[2] = 4.0; a2[3] = 2.0;
    t2[0] = 0.6; t2[1] = 0.0; t2[2] = 0.1;
  // clang-format on

  std::vector<vec_t> data, train;

  for (size_t i = 0; i < 100; i++) {
    data.push_back(a);
    data.push_back(a2);
    train.push_back(t);
    train.push_back(t2);
  }
  optimizer.alpha = 0.1;
  nn.train<mse>(optimizer, data, train, 1, 20);

  vec_t predicted = nn.predict(a);

  EXPECT_NEAR(predicted[0], t[0], 1e-4);
  EXPECT_NEAR(predicted[1], t[1], 1e-4);

  predicted = nn.predict(a2);

  EXPECT_NEAR(predicted[0], t2[0], 1e-4);
  EXPECT_NEAR(predicted[1], t2[1], 1e-4);
}

TEST(recurrent_cell, read_write) {
  recurrent_cell_layer l1(100, 100);
  recurrent_cell_layer l2(100, 100);

  l1.setup(true);
  l2.setup(true);

  serialization_test(l1, l2);
}

TEST(recurrent_cell, forward) {
  recurrent_cell_layer l(4, 2);
  l.weight_init(parameter_init::constant(1.0));
  l.bias_init(parameter_init::constant(0.5));
  l.init_parameters();

  vec_t in = {0, 1, 2, 3};
  std::vector<const Tensor<> *> o;
  l.forward({{Tensor<>(tensor_t{{in}})}}, o);
  vec_t out          = (*o[0]).toTensor()[0];
  vec_t out_expected = {2.5, 2.5};  // 0+1+2+3+0.5

  for (size_t i = 0; i < out_expected.size(); i++) {
    EXPECT_NEAR(out_expected[i], out[i], 1e-4);
  }
}

TEST(recurrent_cell, forward_nobias) {
  recurrent_cell_layer l(4, 2, false);
  l.weight_init(parameter_init::constant(1.0));
  l.init_parameters();

  vec_t in           = {0, 1, 2, 3};
  auto out           = l.forward({{Tensor<>(tensor_t{{in}})}})[0][0].toVec();
  vec_t out_expected = {2.0, 2.0};  // 0+1+2+3

  for (size_t i = 0; i < out_expected.size(); i++) {
    EXPECT_NEAR(out_expected[i], out[i], 1e-4);
  }
}

}  // namespace tiny_dnn
