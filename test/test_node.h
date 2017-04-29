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

  layer_tuple<layer *> tuple_1 = (maxpool, avepool);
  EXPECT_EQ(tuple_1.layers_.at(0)->layer_type(), "max-pool");
  EXPECT_EQ(tuple_1.layers_.at(1)->layer_type(), "ave-pool");

  layer_tuple<layer *> tuple_2 = (maxpool, sgm);
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

  layer_tuple<layer *> tuple_1 = (maxpool, avepool);
  layer_tuple<layer *> tuple_2 = (tuple_1, sgm);

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

  layer_tuple<layer *> tuple_1 = (maxpool, avepool);
  layer_tuple<layer *> tuple_2 = (sgm, tuple_1);

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

TEST(layer_tuple, chain_operator_layer_layer1) {
  fully_connected_layer fc1(2, 2);
  fully_connected_layer fc2(2, 2);
  sigmoid_layer sgm(2);

  fc1 << fc2 << sgm;
  EXPECT_EQ(fc1.next_nodes()[0], &fc2);
  EXPECT_EQ(fc2.next_nodes()[0], &sgm);
}

TEST(layer_tuple, chain_operator_layer_layer2) {
  auto fc1 = std::make_shared<fully_connected_layer>(2, 2);
  auto fc2 = std::make_shared<fully_connected_layer>(2, 2);
  auto sgm = std::make_shared<sigmoid_layer>(2);

  fc1 << fc2 << sgm;
  EXPECT_EQ(fc1->next_nodes()[0], fc2.get());
  EXPECT_EQ(fc2->next_nodes()[0], sgm.get());
}

TEST(layer_tuple, chain_operator_tuple_layer1) {
  fully_connected_layer fc1(2, 2);
  fully_connected_layer fc2(2, 2);
  concat_layer conc({shape3d(2, 1, 1), shape3d(2, 1, 1)});
  sigmoid_layer sgm(4);

  (fc1, fc2) << conc << sgm;
  EXPECT_EQ(conc.prev_nodes()[0], &fc1);
  EXPECT_EQ(conc.prev_nodes()[1], &fc2);
  EXPECT_EQ(conc.next_nodes()[0], &sgm);
}

TEST(layer_tuple, chain_operator_tuple_layer2) {
  auto fc1  = std::make_shared<fully_connected_layer>(2, 2);
  auto fc2  = std::make_shared<fully_connected_layer>(2, 2);
  auto conc = std::make_shared<concat_layer>(
    std::vector<shape3d>({shape3d(2, 1, 1), shape3d(2, 1, 1)}));
  auto sgm = std::make_shared<sigmoid_layer>(4);

  (fc1, fc2) << conc << sgm;
  EXPECT_EQ(conc->prev_nodes()[0], fc1.get());
  EXPECT_EQ(conc->prev_nodes()[1], fc2.get());
  EXPECT_EQ(conc->next_nodes()[0], sgm.get());
}

TEST(layer_tuple, chain_operator_layer_tuple1) {
  convolutional_layer conv(5, 5, 3, 1, 6);
  slice_layer slice(shape3d(3, 3, 6), slice_type::slice_channels, 3);
  tanh_layer tanh(3, 3, 2);
  relu_layer relu(3, 3, 2);
  elu_layer elu(3, 3, 2);

  conv << slice << (tanh, relu, elu);
  EXPECT_EQ(slice.prev_nodes()[0], &conv);
  EXPECT_EQ(slice.next_nodes()[0], &tanh);
  EXPECT_EQ(slice.next_nodes()[1], &relu);
  EXPECT_EQ(slice.next_nodes()[2], &elu);
}

TEST(layer_tuple, chain_operator_layer_tuple2) {
  auto conv  = std::make_shared<convolutional_layer>(5, 5, 3, 1, 6);
  auto slice = std::make_shared<slice_layer>(shape3d(3, 3, 6),
                                             slice_type::slice_channels, 3);
  auto tanh = std::make_shared<tanh_layer>(3, 3, 2);
  auto relu = std::make_shared<relu_layer>(3, 3, 2);
  auto elu  = std::make_shared<elu_layer>(3, 3, 2);

  conv << slice << (tanh, relu, elu);
  EXPECT_EQ(slice->prev_nodes()[0], conv.get());
  EXPECT_EQ(slice->next_nodes()[0], tanh.get());
  EXPECT_EQ(slice->next_nodes()[1], relu.get());
  EXPECT_EQ(slice->next_nodes()[2], elu.get());
}
}  // namespace tiny-dnn
