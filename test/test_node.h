/*
    Copyright (c) 2017, Taiga Nomi
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once
#include "gtest/gtest.h"
#include "testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

namespace tiny_dnn {

TEST(layer_tuple, comma_operator_layer_layer1) {
  max_pooling_layer maxpool(4, 4, 1, 2, 1);
  average_pooling_layer avepool(4, 4, 1, 2, 1);
  sigmoid_layer sgm(4, 4, 1);

  layer_tuple<layerptr_t> tuple_1 = (maxpool, avepool);
  EXPECT_EQ(tuple_1.layers_.at(0)->layer_type(), "max-pool");
  EXPECT_EQ(tuple_1.layers_.at(1)->layer_type(), "ave-pool");

  layer_tuple<layerptr_t> tuple_2 = (maxpool, sgm);
  EXPECT_EQ(tuple_2.layers_.at(0)->layer_type(), "max-pool");
  EXPECT_EQ(tuple_2.layers_.at(1)->layer_type(), "sigmoid-activation");
}

TEST(layer_tuple, comma_operator_layer_layer2) {
  auto maxpool = std::make_shared<max_pooling_layer>(4, 4, 1, 2, 1);
  auto avepool = std::make_shared<average_pooling_layer>(4, 4, 1, 2, 1);
  auto sgm     = std::make_shared<sigmoid_layer>(4, 4, 1);

  layer_tuple<std::shared_ptr<layer>> tuple_1 = (maxpool, avepool);
  EXPECT_EQ(tuple_1.layers_.at(0)->layer_type(), "max-pool");
  EXPECT_EQ(tuple_1.layers_.at(1)->layer_type(), "ave-pool");

  layer_tuple<std::shared_ptr<layer>> tuple_2 = (maxpool, sgm);
  EXPECT_EQ(tuple_2.layers_.at(0)->layer_type(), "max-pool");
  EXPECT_EQ(tuple_2.layers_.at(1)->layer_type(), "sigmoid-activation");
}

TEST(layer_tuple, comma_operator_tuple_layer1) {
  max_pooling_layer maxpool(4, 4, 1, 2, 1);
  average_pooling_layer avepool(4, 4, 1, 2, 1);
  sigmoid_layer sgm(4, 4, 1);

  layer_tuple<layerptr_t> tuple_1 = (maxpool, avepool);
  layer_tuple<layerptr_t> tuple_2 = (tuple_1, sgm);

  EXPECT_EQ(tuple_2.layers_.at(0)->layer_type(), "max-pool");
  EXPECT_EQ(tuple_2.layers_.at(1)->layer_type(), "ave-pool");
  EXPECT_EQ(tuple_2.layers_.at(2)->layer_type(), "sigmoid-activation");
}

TEST(layer_tuple, comma_operator_tuple_layer2) {
  auto maxpool = std::make_shared<max_pooling_layer>(4, 4, 1, 2, 1);
  auto avepool = std::make_shared<average_pooling_layer>(4, 4, 1, 2, 1);
  auto sgm     = std::make_shared<sigmoid_layer>(4, 4, 1);

  layer_tuple<std::shared_ptr<layer>> tuple_1 = (maxpool, avepool);
  layer_tuple<std::shared_ptr<layer>> tuple_2 = (tuple_1, sgm);

  EXPECT_EQ(tuple_2.layers_.at(0)->layer_type(), "max-pool");
  EXPECT_EQ(tuple_2.layers_.at(1)->layer_type(), "ave-pool");
  EXPECT_EQ(tuple_2.layers_.at(2)->layer_type(), "sigmoid-activation");
}

TEST(layer_tuple, comma_operator_layer_tuple1) {
  max_pooling_layer maxpool(4, 4, 1, 2, 1);
  average_pooling_layer avepool(4, 4, 1, 2, 1);
  sigmoid_layer sgm(4, 4, 1);

  layer_tuple<layerptr_t> tuple_1 = (maxpool, avepool);
  layer_tuple<layerptr_t> tuple_2 = (sgm, tuple_1);

  EXPECT_EQ(tuple_2.layers_.at(0)->layer_type(), "sigmoid-activation");
  EXPECT_EQ(tuple_2.layers_.at(1)->layer_type(), "max-pool");
  EXPECT_EQ(tuple_2.layers_.at(2)->layer_type(), "ave-pool");
}

TEST(layer_tuple, comma_operator_layer_tuple2) {
  auto maxpool = std::make_shared<max_pooling_layer>(4, 4, 1, 2, 1);
  auto avepool = std::make_shared<average_pooling_layer>(4, 4, 1, 2, 1);
  auto sgm     = std::make_shared<sigmoid_layer>(4, 4, 1);

  layer_tuple<std::shared_ptr<layer>> tuple_1 = (maxpool, avepool);
  layer_tuple<std::shared_ptr<layer>> tuple_2 = (sgm, tuple_1);

  EXPECT_EQ(tuple_2.layers_.at(0)->layer_type(), "sigmoid-activation");
  EXPECT_EQ(tuple_2.layers_.at(1)->layer_type(), "max-pool");
  EXPECT_EQ(tuple_2.layers_.at(2)->layer_type(), "ave-pool");
}

}  // namespace tiny-dnn
