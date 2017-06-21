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

using namespace tiny_dnn::activation;

namespace tiny_dnn {

TEST(recurrent_cell, train) {
  network<sequential> nn;
  adagrad optimizer;

  nn << recurrent_cell_layer(3, 2, true, new tanh_layer) << sigmoid();

  vec_t a(3), t(2), a2(3), t2(2);

  // clang-format off
    a[0] = 3.0f; a[1] = 0.0f; a[2] = -1.0f;
    t[0] = 0.3f; t[1] = 0.7f;

    a2[0] = 0.2f; a2[1] = 0.5f; a2[2] = 4.0f;
    t2[0] = 0.5f; t2[1] = 0.1f;
  // clang-format on

  std::vector<vec_t> data, train;

  for (int i = 0; i < 100; i++) {
    data.push_back(a);
    data.push_back(a2);
    train.push_back(t);
    train.push_back(t2);
  }
  optimizer.alpha = 0.1f;
  nn.train<mse>(optimizer, data, train, 1, 20);

  vec_t predicted = nn.predict(a);

  EXPECT_NEAR(predicted[0], t[0], 1E-5);
  EXPECT_NEAR(predicted[1], t[1], 1E-5);

  predicted = nn.predict(a2);

  EXPECT_NEAR(predicted[0], t2[0], 1E-5);
  EXPECT_NEAR(predicted[1], t2[1], 1E-5);
}

TEST(recurrent_cell, train_different_batches) {
  auto batch_sizes = {2, 7, 10, 12};
  size_t data_size = std::accumulate(batch_sizes.begin(), batch_sizes.end(), 1,
                                     std::multiplies<int>());
  for (auto &batch_sz : batch_sizes) {
    network<sequential> nn;
    adagrad optimizer;

    nn << recurrent_cell_layer(3, 2) << sigmoid();

    vec_t a(3), t(2), a2(3), t2(2);

    // clang-format off
    a[0] = 3.0f; a[1] = 0.0f; a[2] = -1.0f;
    t[0] = 0.3f; t[1] = 0.7f;

    a2[0] = 0.2f; a2[1] = 0.5f; a2[2] = 4.0f;
    t2[0] = 0.5f; t2[1] = 0.1f;
    // clang-format on

    std::vector<vec_t> data, train;

    for (size_t i = 0; i < data_size; i++) {
      data.push_back(a);
      data.push_back(a2);
      train.push_back(t);
      train.push_back(t2);
    }
    optimizer.alpha = 0.1f;
    nn.train<mse>(optimizer, data, train, batch_sz, 10);

    vec_t predicted = nn.predict(a);

    EXPECT_NEAR(predicted[0], t[0], 1E-5);
    EXPECT_NEAR(predicted[1], t[1], 1E-5);

    predicted = nn.predict(a2);

    EXPECT_NEAR(predicted[0], t2[0], 1E-5);
    EXPECT_NEAR(predicted[1], t2[1], 1E-5);
  }
}

TEST(recurrent_cell, train2) {
  network<sequential> nn;
  gradient_descent optimizer;

  nn << recurrent_cell_layer(4, 6) << tanh() << recurrent_cell_layer(6, 3)
     << tanh();

  vec_t a(4, 0.0), t(3, 0.0), a2(4, 0.0), t2(3, 0.0);

  // clang-format off
    a[0] = 3.0f; a[1] = 1.0f; a[2] = -1.0f; a[3] = 4.0f;
    t[0] = 0.3f; t[1] = 0.7f; t[2] = 0.3f;

    a2[0] = 1.0f; a2[1] = 0.0f; a2[2] = 4.0f; a2[3] = 2.0f;
    t2[0] = 0.6f; t2[1] = 0.0f; t2[2] = 0.1f;
  // clang-format on

  std::vector<vec_t> data, train;

  for (int i = 0; i < 100; i++) {
    data.push_back(a);
    data.push_back(a2);
    train.push_back(t);
    train.push_back(t2);
  }
  optimizer.alpha = 0.1f;
  nn.train<mse>(optimizer, data, train, 1, 20);

  vec_t predicted = nn.predict(a);

  EXPECT_NEAR(predicted[0], t[0], 1E-4);
  EXPECT_NEAR(predicted[1], t[1], 1E-4);

  predicted = nn.predict(a2);

  EXPECT_NEAR(predicted[0], t2[0], 1E-4);
  EXPECT_NEAR(predicted[1], t2[1], 1E-4);
}

TEST(recurrent_cell, gradient_check) {
  network<sequential> nn;
  nn << recurrent_cell_layer(50, 10) << tanh();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second,
                                     epsilon<float_t>(), GRAD_CHECK_ALL));
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
  EXPECT_EQ(l.in_channels(), serial_size_t(7));  // in, h, U, W, V, b and c

  l.weight_init(weight_init::constant(1.0));
  l.bias_init(weight_init::constant(0.5));

  vec_t in = {0, 1, 2, 3};
  std::vector<const tensor_t *> o;
  l.forward({{in}}, o);
  vec_t out          = (*o[0])[0];
  vec_t out_expected = {2.5, 2.5};  // 0+1+2+3+0.5

  for (size_t i = 0; i < out_expected.size(); i++) {
    EXPECT_NEAR(out_expected[i], out[i], 1E-4);
  }
}

TEST(recurrent_cell, forward_nobias) {
  recurrent_cell_layer l(4, 2, false);
  EXPECT_EQ(l.in_channels(), serial_size_t(5));  // in, h, U, W, V, b and c

  l.weight_init(weight_init::constant(1.0));

  vec_t in = {0, 1, 2, 3};
  std::vector<const tensor_t *> o;
  l.forward({{in}}, o);
  vec_t out          = (*o[0])[0];
  vec_t out_expected = {2.0, 2.0};  // 0+1+2+3

  for (size_t i = 0; i < out_expected.size(); i++) {
    EXPECT_NEAR(out_expected[i], out[i], 1E-4);
  }
}

}  // namespace tiny_dnn
