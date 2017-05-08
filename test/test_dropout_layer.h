/*
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once
#include <deque>
#include <vector>
#include "gtest/gtest.h"
#include "testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn::activation;

namespace tiny_dnn {

TEST(dropout, randomized) {
  int num_units                  = 10000;
  tiny_dnn::float_t dropout_rate = 0.1f;
  dropout_layer l(num_units, dropout_rate, net_phase::train);
  vec_t v(num_units, 1.0);

  l.forward({{v}});
  const auto mask1 = l.get_mask(0);

  l.forward({{v}});
  const auto mask2 = l.get_mask(0);

  // mask should change for each fprop
  EXPECT_TRUE(is_different_container(mask1, mask2));

  // dropout-rate should be around 0.1
  double margin_factor = 0.9;
  int64_t num_on1      = std::count(mask1.begin(), mask1.end(), 1);
  int64_t num_on2      = std::count(mask2.begin(), mask2.end(), 1);

  EXPECT_LE(num_units * dropout_rate * margin_factor, num_on1);
  EXPECT_GE(num_units * dropout_rate / margin_factor, num_on1);
  EXPECT_LE(num_units * dropout_rate * margin_factor, num_on2);
  EXPECT_GE(num_units * dropout_rate / margin_factor, num_on2);
}

TEST(dropout, read_write) {
  dropout_layer l1(1024, 0.5, net_phase::test);
  dropout_layer l2(1024, 0.5, net_phase::test);

  l1.init_weight();
  l2.init_weight();

  serialization_test(l1, l2);
}

TEST(dropout, full_net) {
  network<sequential> nn;
  adam optimizer;

  vec_t a(4, 0.0), t(2, 0.0), a2(4, 0.0), t2(2, 0.0);

  // clang-format off
  a[0] = 3.0f; a[1] = 1.0f; a[2] = -1.0f; a[3] = 4.0f;
  t[0] = 0.3f; t[1] = 0.7f;

  a2[0] = 1.0f; a2[1] = 0.0f; a2[2] = 4.0f; a2[3] = 2.0f;
  t2[0] = 0.6f; t2[1] = 0.0f;
  // clang-format on

  std::vector<vec_t> data, train;

  for (int i = 0; i < 100; i++) {
    data.push_back(a);
    data.push_back(a2);
    train.push_back(t);
    train.push_back(t2);
  }

  nn << fully_connected_layer(4, 10) << relu() << dropout(10, 0.5)
     << fully_connected_layer(10, 2) << sigmoid();

  nn.train<mse>(optimizer, data, train, 1, 10);
  // batch = 11,20,50
}

TEST(dropout, full_net_batch) {
  network<sequential> nn;
  adam optimizer;

  vec_t a(4, 0.0), t(2, 0.0), a2(4, 0.0), t2(2, 0.0);

  // clang-format off
  a[0] = 3.0f; a[1] = 1.0f; a[2] = -1.0f; a[3] = 4.0f;
  t[0] = 0.3f; t[1] = 0.7f;

  a2[0] = 1.0f; a2[1] = 0.0f; a2[2] = 4.0f; a2[3] = 2.0f;
  t2[0] = 0.6f; t2[1] = 0.0f;
  // clang-format on

  std::vector<vec_t> data, train;

  for (int i = 0; i < 100; i++) {
    data.push_back(a);
    data.push_back(a2);
    train.push_back(t);
    train.push_back(t2);
  }

  nn << fully_connected_layer(4, 10) << relu() << dropout(10, 0.5)
     << fully_connected_layer(10, 2) << sigmoid();

  nn.train<mse>(optimizer, data, train, 20, 10);
}
}  // namespace tiny-dnn
