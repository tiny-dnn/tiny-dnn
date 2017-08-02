/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <gtest/gtest.h>

#include "test/testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

namespace tiny_dnn {

TEST(selu, gradient_check) {
  network<sequential> nn;
  nn << selu(size_t{3}, size_t{3}, size_t{1});

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_parameters();
  EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second,
                                     epsilon<float_t>(), GRAD_CHECK_ALL));
}
}  // namespace tiny_dnn
