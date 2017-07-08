/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once
#include "gtest/gtest.h"
#include "testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

namespace tiny_dnn {

using namespace tiny_dnn::activation;

TEST(fully_connected, setup_internal) {
  fully_connected_layer l(32, 10, true, core::backend_t::internal);

  EXPECT_EQ(l.parallelize(), true);            // if layer can be parallelized
  EXPECT_EQ(l.in_channels(), 1u);              // num of input tensors
  EXPECT_EQ(l.out_channels(), 1u);             // num of output tensors
  EXPECT_EQ(l.in_data_size(), 32u);            // size of input tensors
  EXPECT_EQ(l.out_data_size(), 10u);           // size of output tensors
  EXPECT_EQ(l.fan_in_size(), 32u);             // num of incoming connections
  EXPECT_EQ(l.fan_out_size(), 10u);            // num of outgoing connections
  EXPECT_EQ(l.parameters().size(), 2u);        // num of trainable parameters
  EXPECT_EQ(l.ith_parameter(0).size(), 320u);  // size of weight parameter
  EXPECT_EQ(l.ith_parameter(1).size(), 10u);   // size of bias parameter
  EXPECT_STREQ(l.layer_type().c_str(),
               "fully-connected");  // string with layer type
}

TEST(fully_connected, forward) {
  fully_connected_layer l(3, 2, true);

  vec_t in = {1, 2, 3};
  // clang-format off
  vec_t weights = {
    2, 6, 4,
    4, 6, 2
  };
  vec_t bias = {-4, 22};
  // clang-format on
  vec_t expected = {24, 42};

  l.ith_parameter(0).set_data(Tensor<float_t>(weights));
  l.ith_parameter(1).set_data(Tensor<float_t>(bias));

  std::vector<const tensor_t *> out;
  l.forward({{in}}, out);
  vec_t result = (*out[0])[0];

  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_FLOAT_EQ(expected[i], result[i]);
  }
}

TEST(fully_connected, train) {
  network<sequential> nn;
  adagrad optimizer;

  nn << fully_connected_layer(3, 2) << sigmoid();

  vec_t a(3), t(2), a2(3), t2(2);

  // clang-format off
    a[0] = 3.0; a[1] = 0.0; a[2] = -1.0;
    t[0] = 0.3; t[1] = 0.7;

    a2[0] = 0.2; a2[1] = 0.5; a2[2] = 4.0;
    t2[0] = 0.5; t2[1] = 0.1;
  // clang-format on

  std::vector<vec_t> data, train;

  for (int i = 0; i < 100; i++) {
    data.push_back(a);
    data.push_back(a2);
    train.push_back(t);
    train.push_back(t2);
  }
  optimizer.alpha = 0.1;
  nn.train<mse>(optimizer, data, train, 1, 10);

  vec_t predicted = nn.predict(a);

  EXPECT_NEAR(predicted[0], t[0], 1E-5);
  EXPECT_NEAR(predicted[1], t[1], 1E-5);

  predicted = nn.predict(a2);

  EXPECT_NEAR(predicted[0], t2[0], 1E-5);
  EXPECT_NEAR(predicted[1], t2[1], 1E-5);
}

/* todo (karandesai) : Inspect problem, there is a major precision loss
TEST(fully_connected, train_different_batches) {
  auto batch_sizes = {2, 7, 10, 12};
  size_t data_size = std::accumulate(batch_sizes.begin(), batch_sizes.end(), 1,
                                     std::multiplies<int>());
  for (auto &batch_sz : batch_sizes) {
    network<sequential> nn;
    adagrad optimizer;

    nn << fully_connected_layer(3, 2) << sigmoid();

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

    EXPECT_NEAR(predicted[0], t[0], 1E-5);
    EXPECT_NEAR(predicted[1], t[1], 1E-5);

    predicted = nn.predict(a2);

    EXPECT_NEAR(predicted[0], t2[0], 1E-5);
    EXPECT_NEAR(predicted[1], t2[1], 1E-5);
  }
}
*/

TEST(fully_connected, train2) {
  network<sequential> nn;
  gradient_descent optimizer;

  nn << fully_connected_layer(4, 6) << tanh() << fully_connected_layer(6, 3)
     << tanh();

  vec_t a(4, 0.0), t(3, 0.0), a2(4, 0.0), t2(3, 0.0);

  // clang-format off
    a[0] = 3.0; a[1] = 1.0; a[2] = -1.0; a[3] = 4.0;
    t[0] = 0.3; t[1] = 0.7; t[2] = 0.3;

    a2[0] = 1.0; a2[1] = 0.0; a2[2] = 4.0; a2[3] = 2.0;
    t2[0] = 0.6; t2[1] = 0.0; t2[2] = 0.1;
  // clang-format on

  std::vector<vec_t> data, train;

  for (int i = 0; i < 100; i++) {
    data.push_back(a);
    data.push_back(a2);
    train.push_back(t);
    train.push_back(t2);
  }
  optimizer.alpha = 0.1;
  nn.train<mse>(optimizer, data, train, 1, 10);

  vec_t predicted = nn.predict(a);

  EXPECT_NEAR(predicted[0], t[0], 1E-4);
  EXPECT_NEAR(predicted[1], t[1], 1E-4);

  predicted = nn.predict(a2);

  EXPECT_NEAR(predicted[0], t2[0], 1E-4);
  EXPECT_NEAR(predicted[1], t2[1], 1E-4);
}

TEST(fully_connected, gradient_check) {
  network<sequential> nn;
  nn << fully_connected_layer(50, 10) << tanh();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second,
                                     epsilon<float_t>(), GRAD_CHECK_ALL));
}

/* todo (karandesai) : deal with serialization after parameter integration later
 * uncomment after fixing
TEST(fully_connected, read_write) {
  fully_connected_layer l1(100, 100);
  fully_connected_layer l2(100, 100);

  l1.setup(true);
  l2.setup(true);

  serialization_test(l1, l2);
}
*/

#ifdef CNN_USE_NNPACK
TEST(fully_connected, forward_nnp) {
  fully_connected_layer l(4, 2, true, core::backend_t::nnpack);
  l.weight_init_f(parameter_init::constant(1.0));
  l.bias_init_f(parameter_init::constant(0.5));

  vec_t in = {0, 1, 2, 3};
  std::vector<const tensor_t *> o;
  l.forward({{in}}, o);
  vec_t out          = (*o[0])[0];
  vec_t out_expected = {6.5, 6.5};  // 0+1+2+3+0.5

  for (size_t i = 0; i < out_expected.size(); i++) {
    EXPECT_FLOAT_EQ(out_expected[i], out[i]);
  }
}

TEST(fully_connected, forward_nnp_nobias) {
  fully_connected_layer l(4, 2, false, core::backend_t::nnpack);
  l.weight_init_f(parameter_init::constant(1.0));

  vec_t in = {0, 1, 2, 3};
  std::vector<const tensor_t *> o;
  l.forward({{in}}, o);
  vec_t out          = (*o[0])[0];
  vec_t out_expected = {6.0, 6.0};  // 0+1+2+3

  for (size_t i = 0; i < out_expected.size(); i++) {
    EXPECT_FLOAT_EQ(out_expected[i], out[i]);
  }
}
#endif

TEST(fully_connected, forward_nobias) {
  fully_connected_layer l(4, 2, false);
  l.weight_init_f(parameter_init::constant(1.0));

  vec_t in = {0, 1, 2, 3};
  std::vector<const tensor_t *> o;
  l.forward({{in}}, o);
  vec_t out          = (*o[0])[0];
  vec_t out_expected = {6.0, 6.0};  // 0+1+2+3

  for (size_t i = 0; i < out_expected.size(); i++) {
    EXPECT_FLOAT_EQ(out_expected[i], out[i]);
  }
}

}  // namespace tiny_dnn
