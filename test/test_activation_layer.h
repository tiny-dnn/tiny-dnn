#pragma once
#include "gtest/gtest.h"
#include "testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn::activation;

namespace tiny_dnn {
TEST(selu, gradient_check) {
  network<sequential> nn;
  nn << selu(size_t{3}, size_t{3}, size_t{1});

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second,
                                     epsilon<float_t>(), GRAD_CHECK_ALL));
}
}  // namespace tiny_dnn
