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

using namespace tiny_dnn::activation;

namespace tiny_dnn {

TEST(fully_connected, forward) {
  fully_connected_layer l(4, 2);
  EXPECT_EQ(l.in_channels(), size_t(3));  // in, W and b

  l.weight_init(weight_init::constant(1.0));
  l.bias_init(weight_init::constant(0.5));

  vec_t in = {0, 1, 2, 3};
  std::vector<const tensor_t *> o;
  l.forward({{in}}, o);
  vec_t out          = (*o[0])[0];
  vec_t out_expected = {6.5, 6.5};  // 0+1+2+3+0.5

  for (size_t i = 0; i < out_expected.size(); i++) {
    EXPECT_FLOAT_EQ(out_expected[i], out[i]);
  }
}

#ifdef CNN_USE_NNPACK
TEST(fully_connected, forward_nnp) {
  nnp_initialize();
  fully_connected_layer l(4, 2, true, core::backend_t::nnpack);
  EXPECT_EQ(l.in_channels(), size_t(3));  // in, W and b

  l.weight_init(weight_init::constant(1.0));
  l.bias_init(weight_init::constant(0.5));

  vec_t in           = {0, 1, 2, 3};
  vec_t out          = l.forward({{in}})[0][0];
  vec_t out_expected = {6.5, 6.5};  // 0+1+2+3+0.5

  for (size_t i = 0; i < out_expected.size(); i++) {
    EXPECT_FLOAT_EQ(out_expected[i], out[i]);
  }
}
#endif

TEST(fully_connected, forward_nobias) {
  fully_connected_layer l(4, 2, false);
  EXPECT_EQ(l.in_channels(), 2u);  // in and W

  l.weight_init(weight_init::constant(1.0));

  vec_t in = {0, 1, 2, 3};
  std::vector<const tensor_t *> o;
  l.forward({{in}}, o);
  vec_t out          = (*o[0])[0];
  vec_t out_expected = {6.0, 6.0};  // 0+1+2+3

  for (size_t i = 0; i < out_expected.size(); i++) {
    EXPECT_FLOAT_EQ(out_expected[i], out[i]);
  }
}

TEST(fully_connected, serialization) {
  fully_connected_layer l1(100, 100);
  fully_connected_layer l2(100, 100);

  l1.setup(true);
  l2.setup(true);

  serialization_test(l1, l2);
}


}  // namespace tiny_dnn
