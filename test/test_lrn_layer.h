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

namespace tiny_dnn {

TEST(lrn, cross) {
  lrn_layer lrn(1, 1, 3, 4, /*alpha=*/1.5, /*beta=*/2.0,
                norm_region::across_channels);

  tiny_dnn::float_t in[4]       = {-1.0, 3.0, 2.0, 5.0};
  tiny_dnn::float_t expected[4] = {
    -1.0f / 36.0f,        // -1.0 / (1+0.5*(1*1+3*3))^2
    3.0f / 64.0f,         //  3.0 / (1+0.5*(1*1+3*3+2*2))^2
    2.0f / 400.0f,        //  2.0 / (1+0.5*(3*3+2*2+5*5))^2
    5.0f / 15.5f / 15.5f  // 5.0 / (1+0.5*(2*2+5*5))^2
  };

  auto out = lrn.forward({{vec_t(in, in + 4)}})[0][0];

  EXPECT_NEAR(expected[0], out[0], epsilon<float_t>());
  EXPECT_NEAR(expected[1], out[1], epsilon<float_t>());
  EXPECT_NEAR(expected[2], out[2], epsilon<float_t>());
  EXPECT_NEAR(expected[3], out[3], epsilon<float_t>());
}

TEST(lrn, read_write) {
  lrn_layer l1(10, 10, 3, 4, 1.5f, 2.0f, norm_region::across_channels);
  lrn_layer l2(10, 10, 3, 4, 1.5f, 2.0f, norm_region::across_channels);

  l1.init_weight();
  l2.init_weight();

  serialization_test(l1, l2);
}

}  // namespace tiny-dnn
