/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <gtest/gtest.h>
#include <string>

#include "test/testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

namespace tiny_dnn {

TEST(serialization_hdf, load_parameter) {
  std::string xor_hdf_path;
  resolve_path("testdata/xor.h5", xor_hdf_path);

  size_t in_dim(2), out_dim(4);
  Parameter dense_1_W(1, 1, out_dim, in_dim, parameter_type::weight);
  Parameter dense_1_b(1, 1, 1, out_dim, parameter_type::bias);
  dense_1_W.load(xor_hdf_path, "dense_1/dense_1/kernel:0");
  dense_1_b.load(xor_hdf_path, "dense_1/dense_1/bias:0");

  // clang-format off
  Tensor<> expected_W{vec_t{ 1.7683, -0.6835, -0.4390, -1.6895,
                            -1.7678, -0.2821, -0.2773,  1.6886}};
  // clang-format on

  Tensor<> expected_b{vec_t{0., 0., 0., -0.0012}};

  for (size_t i = 0; i < expected_W.size(); i++) {
    EXPECT_NEAR(expected_W.host_at(i), *(dense_1_W.data_at(i)), 1E-3);
  }
  for (size_t i = 0; i < expected_b.size(); i++) {
    EXPECT_NEAR(expected_b.host_at(i), *(dense_1_b.data_at(i)), 1E-3);
  }
}

}  // namespace tiny_dnn
