/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <functional>
#include <vector>

#include "tiny_dnn/util/gradient_check.h"

namespace tiny_dnn {

// fully connected
TEST(integration, train1) {
  network<sequential> nn;
  adagrad optimizer;

  nn << recurrent_layer(rnn(3, 20), 1) << fully_connected_layer(20, 2)
     << sigmoid();
  nn.weight_init(weight_init::xavier());
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

  EXPECT_NEAR(predicted[0], t[0], 1e-5);
  EXPECT_NEAR(predicted[1], t[1], 1e-5);

  predicted = nn.predict(a2);

  EXPECT_NEAR(predicted[0], t2[0], 1e-5);
  EXPECT_NEAR(predicted[1], t2[1], 1e-5);
}

TEST(integration, train_different_batches1) {
  auto batch_sizes = {2, 7, 10, 12};
  size_t data_size = std::accumulate(batch_sizes.begin(), batch_sizes.end(), 1,
                                     std::multiplies<int>());
  for (auto &batch_sz : batch_sizes) {
    network<sequential> nn;
    adagrad optimizer;

    nn << fully_connected_layer(3, 2) << sigmoid();
    nn.weight_init(weight_init::xavier());

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

TEST(integration, train2) {
  network<sequential> nn;
  gradient_descent optimizer;

  nn << fully_connected_layer(4, 6) << selu() << fully_connected_layer(6, 3)
     << tanh_layer();

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

  EXPECT_NEAR(predicted[0], t[0], 1e-4);
  EXPECT_NEAR(predicted[1], t[1], 1e-4);

  predicted = nn.predict(a2);

  EXPECT_NEAR(predicted[0], t2[0], 1e-4);
  EXPECT_NEAR(predicted[1], t2[1], 1e-4);
}

TEST(integration, gradient_check1) {
  network<sequential> nn;
  nn << fully_connected_layer(50, 10) << tanh_layer();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second,
                                     epsilon<float_t>(), GRAD_CHECK_ALL));
}

// convolutional integration
TEST(integratoin, gradient_check2) {  // tanh - mse
  network<sequential> nn;
  nn << convolutional_layer(5, 5, 3, 1, 1) << activation::tanh();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second,
                                     epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(integration, gradient_check3) {  // sigmoid - mse
  network<sequential> nn;
  nn << convolutional_layer(5, 5, 3, 1, 1) << sigmoid();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second,
                                     epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(integration, gradient_check4) {  // rectified - mse
  network<sequential> nn;

  nn << convolutional_layer(5, 5, 3, 1, 1) << relu();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second,
                                     epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(integration, gradient_check5) {  // identity - mse
  network<sequential> nn;

  nn << convolutional_layer(5, 5, 3, 1, 1);

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second,
                                     epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(integration, gradient_check6) {  // sigmoid - cross-entropy
  network<sequential> nn;

  nn << convolutional_layer(5, 5, 3, 1, 1) << sigmoid();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<cross_entropy>(
    test_data.first, test_data.second, epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(integration, gradient_check7) {  // sigmoid - absolute
  network<sequential> nn;

  nn << convolutional_layer(5, 5, 3, 1, 1) << sigmoid();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<absolute>(test_data.first, test_data.second,
                                          epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(integration, gradient_check8) {  // sigmoid - absolute eps
  network<sequential> nn;

  nn << convolutional_layer(5, 5, 3, 1, 1) << sigmoid();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<absolute_eps<100>>(
    test_data.first, test_data.second, epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(integration, gradient_check9_pad_same) {  // sigmoid - mse - padding same
  network<sequential> nn;

  nn << convolutional_layer(5, 5, 3, 1, 1, padding::same, true, 1, 1, 1, 1,
                            core::backend_t::internal)
     << sigmoid();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second,
                                     epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(integration, gradient_check10_w_stride) {  // sigmoid - mse - w_stride > 1
  network<sequential> nn;

  nn << convolutional_layer(3, 3, 1, 1, 1, padding::valid, true, 2, 1, 1, 1,
                            core::backend_t::internal)
     << sigmoid();

  const auto test_data = generate_gradient_check_data(nn.in_data_size(), 1);
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second,
                                     epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(integration, gradient_check11_h_stride) {  // sigmoid - mse - h_stride > 1
  network<sequential> nn;

  nn << convolutional_layer(3, 3, 1, 1, 1, padding::valid, true, 1, 2, 1, 1,
                            core::backend_t::internal)
     << sigmoid();

  const auto test_data = generate_gradient_check_data(nn.in_data_size(), 1);
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second,
                                     epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(integration,
     gradient_check12_connection_tbl) {  // sigmoid - mse - has connection-tbl
  network<sequential> nn;
  bool tbl[3 * 3] = {true, false, true, false, true, false, true, true, false};

  core::connection_table connections(tbl, 3, 3);

  nn << convolutional_layer(7, 7, 3, 3, 1, connections, padding::valid, true, 1,
                            1, 1, 1, core::backend_t::internal)
     << sigmoid();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second,
                                     epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(integration, gradient_check13_pad_same) {  // sigmoid - mse - padding same
  network<sequential> nn;

  nn << fully_connected_layer(10, 5 * 5)
     << convolutional_layer(5, 5, 3, 1, 1, padding::same, true, 1, 1, 1, 1,
                            core::backend_t::internal)
     << sigmoid();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second,
                                     epsilon<float_t>(), GRAD_CHECK_ALL));
}

// power layer
TEST(integration, gradient_check14) {
  network<sequential> nn;

  nn << fully_connected_layer(10, 20) << tanh_layer()
     << power_layer(shape3d(20, 1, 1), 3.0, 1.5)
     << fully_connected_layer(20, 10) << tanh_layer();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second,
                                     epsilon<float_t>(), GRAD_CHECK_ALL));
}

// selu
TEST(integration, gradient_check15) {
  network<sequential> nn;
  nn << selu(size_t{3}, size_t{3}, size_t{1});

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second,
                                     epsilon<float_t>(), GRAD_CHECK_ALL));
}

// ave pooling
TEST(integration, gradient_check16) {  // sigmoid - cross-entropy
  using loss_func  = cross_entropy;
  using activation = sigmoid;
  using network    = network<sequential>;

  network nn;
  nn << fully_connected_layer(3, 8) << activation()
     << average_pooling_layer(4, 2, 1, 2) << activation();  // 4x2 => 2x1

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();

  EXPECT_TRUE(nn.gradient_check<loss_func>(test_data.first, test_data.second,
                                           epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(integration, gradient_check17) {  // x-stride
  using loss_func  = cross_entropy;
  using activation = sigmoid;
  using network    = network<sequential>;

  network nn;
  nn << fully_connected_layer(3, 8) << activation()
     << average_pooling_layer(4, 2, 1, 2, 1, 2, 1, false)  // 4x2 => 2x2
     << activation();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();

  EXPECT_TRUE(nn.gradient_check<loss_func>(test_data.first, test_data.second,
                                           epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(integration, gradient_check18) {  // y-stride
  using loss_func  = cross_entropy;
  using activation = sigmoid;
  using network    = network<sequential>;

  network nn;
  nn << fully_connected_layer(3, 8) << activation()
     << average_pooling_layer(4, 2, 1, 1, 2, 1, 2, false)  // 4x2 => 4x1
     << activation();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();

  EXPECT_TRUE(nn.gradient_check<loss_func>(test_data.first, test_data.second,
                                           epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(integration, gradient_check19) {  // padding-same
  using loss_func  = cross_entropy;
  using activation = sigmoid;
  using network    = network<sequential>;

  network nn;
  nn << average_pooling_layer(4, 2, 1, 2, 2, 1, 1, false, padding::same)
     << activation();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();

  EXPECT_TRUE(nn.gradient_check<loss_func>(test_data.first, test_data.second,
                                           epsilon<float_t>(), GRAD_CHECK_ALL));
}

}  // namespace tiny_dnn
