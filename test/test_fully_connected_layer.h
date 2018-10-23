/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <functional>
#include <vector>

namespace tiny_dnn {

TEST(fully_connected, gradient_check) {
  const size_t in_size  = 50;
  const size_t out_size = 10;
  fully_connected_layer fc(in_size, out_size);
  std::vector<tensor_t> input_data =
    generate_test_data({1, 1, 1}, {in_size, in_size * out_size, out_size});
  std::vector<tensor_t> in_grad  = input_data;  // copy constructor
  std::vector<tensor_t> out_data = generate_test_data({1}, {out_size});
  std::vector<tensor_t> out_grad = generate_test_data({1}, {out_size});
  const size_t trials            = 100;
  for (size_t i = 0; i < trials; i++) {
    const size_t in_edge  = uniform_idx(input_data);
    const size_t in_idx   = uniform_idx(input_data[in_edge][0]);
    const size_t out_edge = uniform_idx(out_data);
    const size_t out_idx  = uniform_idx(out_data[out_edge][0]);
    float_t ngrad = numeric_gradient(fc, input_data, in_edge, in_idx, out_data,
                                     out_edge, out_idx);
    float_t cgrad = analytical_gradient(fc, input_data, in_edge, in_idx,
                                        out_data, out_grad, out_edge, out_idx);
    EXPECT_NEAR(ngrad, cgrad, epsilon<float_t>());
  }
}

TEST(fully_connected, read_write) {
  fully_connected_layer l1(100, 100);
  fully_connected_layer l2(100, 100);

  l1.setup(true);
  l2.setup(true);

  serialization_test(l1, l2);
}

TEST(fully_connected, forward) {
  fully_connected_layer l(4, 2);
  EXPECT_EQ(l.in_channels(), 3u);  // in, W and b

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

void test_fully_connected_forward(core::backend_t backend) {
  fully_connected_layer l(4, 2, true, backend);
  EXPECT_EQ(l.in_channels(), 3u);  // in, W and b

  l.weight_init(weight_init::constant(1.0));
  l.bias_init(weight_init::constant(0.5));

  vec_t in                  = {0, 1, 2, 3};
  std::vector<const tensor_t*> tout;
  l.forward({{in}}, tout);
  vec_t out                 = (*tout[0])[0];
  vec_t out_expected        = {6.5, 6.5};  // 0+1+2+3+0.5

  for (size_t i = 0; i < out_expected.size(); i++) {
    EXPECT_FLOAT_EQ(out_expected[i], out[i]);
  }
}

#ifdef CNN_USE_NNPACK
TEST(fully_connected, forward_nnp) {
  nnp_initialize();
  test_fully_connected_forward(core::backend_t::nnpack);
}
#endif

#ifdef CNN_USE_INTEL_MKL
TEST(fully_connected, forward_intel_mkl) {
  test_fully_connected_forward(core::backend_t::intel_mkl);
}
#endif

#ifdef CNN_USE_AVX
TEST(fully_connected, forward_avx) {
  test_fully_connected_forward(core::backend_t::avx);
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

}  // namespace tiny_dnn
