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

TEST(parameter, init) {
  Parameter p(1, 1, 3, 3, parameter_type::weight, true);

  EXPECT_EQ(p.shape().width_, 3u);
  EXPECT_EQ(p.shape().height_, 3u);
  EXPECT_EQ(p.shape().depth_, 1u);
  EXPECT_EQ(p.size(), 9u);
  EXPECT_EQ(p.type(), parameter_type::weight);

  EXPECT_TRUE(p.is_trainable());
}

TEST(parameter, getter_setter) {
  Parameter p(4, 1, 1, 1, parameter_type::bias, false);
  Tensor<float_t> t{{1.0, 2.0, 3.0, 4.0}};

  p.set_data(t);
  Tensor<float_t> *pt = p.data();

  for (size_t i = 0; i < t.size(); i++) {
    EXPECT_EQ(pt->host_at(i), t.host_at(i));
  }
}

TEST(parameter, merge_grads) {
  Tensor<float_t> grad0{tensor_t{{1.0, 2.0}, {2.0, 1.0}}};
  Tensor<float_t> gradp{tensor_t{{2.0, 4.0}, {4.0, 2.0}}};

  Parameter p(2, 1, 1, 1, parameter_type::bias, false);
  p.set_grad(gradp);
  p.merge_grads(&grad0);

  Tensor<float_t> expected{tensor_t{{6.0, 6.0}, {6.0, 6.0}}};

  for (size_t i = 0; i < p.size(); i++) {
    EXPECT_EQ(grad0.host_at(0, i), expected.host_at(0, i));
    EXPECT_EQ(grad0.host_at(1, i), expected.host_at(1, i));
  }
}

TEST(parameter, layer_adder) {
  fully_connected_layer fc(3, 2);

  // todo (karandesai): modify later, two parameters will already be there
  // in fc layer
  fc.add_parameter(3, 2, 1, 1, parameter_type::weight);
  fc.add_parameter(2, 1, 1, 1, parameter_type::bias);

  auto parameters = fc.parameters();

  // check whether they were added in proper order
  ASSERT_GE(parameters.size(), 2);
  EXPECT_EQ(parameters[0]->type(), parameter_type::weight);
  EXPECT_EQ(parameters[1]->type(), parameter_type::bias);
}

// todo (karandesai) : test getters and setters on fc layer

}  // namespace tiny_dnn
