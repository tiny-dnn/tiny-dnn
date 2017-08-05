/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <gtest/gtest.h>

#include <vector>

#include "test/testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

namespace tiny_dnn {

TEST(lrn, cross) {
  lrn_layer lrn(1, 1, 3, 4, /*alpha=*/1.5, /*beta=*/2.0,
                norm_region::across_channels);

  tiny_dnn::float_t in[4]       = {-1.0, 3.0, 2.0, 5.0};
  tiny_dnn::float_t expected[4] = {
    -1.0 / 36.0,       // -1.0 / (1+0.5*(1*1+3*3))^2
    3.0 / 64.0,        //  3.0 / (1+0.5*(1*1+3*3+2*2))^2
    2.0 / 400.0,       //  2.0 / (1+0.5*(3*3+2*2+5*5))^2
    5.0 / 15.5 / 15.5  //  5.0 / (1+0.5*(2*2+5*5))^2
  };
  std::vector<const Tensor<>*> o;
  lrn.forward({{Tensor<>(vec_t(in, in + 4))}}, o);
  auto out = (*o[0]).toTensor()[0];

  EXPECT_NEAR(expected[0], out[0], epsilon<float_t>());
  EXPECT_NEAR(expected[1], out[1], epsilon<float_t>());
  EXPECT_NEAR(expected[2], out[2], epsilon<float_t>());
  EXPECT_NEAR(expected[3], out[3], epsilon<float_t>());
}

TEST(lrn, read_write) {
  lrn_layer l1(10, 10, 3, 4, 1.5, 2.0, norm_region::across_channels);
  lrn_layer l2(10, 10, 3, 4, 1.5, 2.0, norm_region::across_channels);

  l1.init_weight();
  l2.init_weight();

  serialization_test(l1, l2);
}

}  // namespace tiny_dnn
