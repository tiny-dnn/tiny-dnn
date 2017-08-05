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
TEST(deconvolutional, setup_tiny) {
  deconvolutional_layer l(2, 2, 3, 1, 2, padding::valid, true, 1, 1,
                          core::backend_t::internal);

  EXPECT_EQ(l.parallelize(), true);           // if layer can be parallelized
  EXPECT_EQ(l.in_channels(), 1u);             // num of input tensors
  EXPECT_EQ(l.out_channels(), 1u);            // num of output tensors
  EXPECT_EQ(l.in_data_size(), 4u);            // size of input tensors
  EXPECT_EQ(l.out_data_size(), 32u);          // size of output tensors
  EXPECT_EQ(l.in_data_shape().size(), 1u);    // number of inputs shapes
  EXPECT_EQ(l.out_data_shape().size(), 1u);   // num of output shapes
  EXPECT_EQ(l.inputs().size(), 1u);           // num of input edges
  EXPECT_EQ(l.outputs().size(), 1u);          // num of outpus edges
  EXPECT_EQ(l.in_types().size(), 1u);         // num of input data types
  EXPECT_EQ(l.out_types().size(), 1u);        // num of output data types
  EXPECT_EQ(l.fan_in_size(), 9u);             // num of incoming connections
  EXPECT_EQ(l.fan_out_size(), 18u);           // num of outgoing connections
  EXPECT_EQ(l.parameters().size(), 2u);       // num of trainable parameters
  EXPECT_EQ(l.weights_at()[0]->size(), 18u);  // size of weight parameter
  EXPECT_EQ(l.bias_at()[0]->size(), 2u);      // size of bias parameter
  EXPECT_STREQ(l.layer_type().c_str(), "deconv");  // string with layer type
}

#ifdef CNN_USE_NNPACK
TEST(deconvolutional, setup_nnp) {
  deconvolutional_layer l(2, 2, 3, 1, 2, padding::valid, true, 1, 1,
                          backend_t::nnpack);

  EXPECT_EQ(l.parallelize(), true);          // if layer can be parallelized
  EXPECT_EQ(l.in_channels(), 1u);            // num of input tensors
  EXPECT_EQ(l.out_channels(), 1u);           // num of output tensors
  EXPECT_EQ(l.in_data_size(), 4u);           // size of input tensors
  EXPECT_EQ(l.out_data_size(), 32u);         // size of output tensors
  EXPECT_EQ(l.in_data_shape().size(), 1u);   // number of inputs shapes
  EXPECT_EQ(l.out_data_shape().size(), 1u);  // num of output shapes
  EXPECT_EQ(l.inputs().size(), 1u);          // num of input edges
  EXPECT_EQ(l.outputs().size(), 1u);         // num of outpus edges
  EXPECT_EQ(l.in_types().size(), 1u);        // num of input data types
  EXPECT_EQ(l.out_types().size(), 1u);       // num of output data types
  EXPECT_EQ(l.fan_in_size(), 9u);            // num of incoming connections
  EXPECT_EQ(l.fan_out_size(), 18u);          // num of outgoing connections
  EXPECT_EQ(l.parameters().size(), 2u);      // num of trainable parameters
  EXPECT_EQ(l.weights_at().size(), 18u);     // size of weight parameter
  EXPECT_EQ(l.bias_at().size(), 2u);         // size of bias parameter
  EXPECT_STREQ(l.layer_type().c_str(), "deconv");  // string with layer type
}
#endif

TEST(deconvolutional, fprop) {
  deconvolutional_layer l(2, 2, 3, 1, 2);
  l.weight_init_f(parameter_init::constant(0.0));

  vec_t in(4, 0);
  vec_t out_expected(32, 0);
  uniform_rand(in.begin(), in.end(), -1.0, 1.0);

  auto out         = l.forward({{Tensor<>(in)}});
  vec_t out_result = (*out[0]).toTensor()[0];

  for (size_t i = 0; i < out_result.size(); i++) {
    EXPECT_FLOAT_EQ(out_result[i], float_t{0});
  }

  // clang-format off
  vec_t weight = {
      0.3,  0.1,  0.2,
      0.0, -0.1, -0.1,
      0.05, -0.2,  0.05,

      0.0, -0.1,  0.1,
      0.1, -0.2,  0.3,
      0.2, -0.3,  0.2
  };

  in = {
      3.0, 2.0,
      3.0, 0.0
  };

  out_expected = {
      0.9,  0.9,  0.8,  0.4,
      0.9,  0.0,  0.1, -0.2,
      0.15, -0.80, -0.55,  0.1,
      0.15, -0.60,  0.15,  0.0,

      0.0, -0.3,  0.1,  0.2,
      0.3, -0.7,  0.8,  0.6,
      0.9, -1.1,  0.9,  0.4,
      0.6, -0.9,  0.6,  0.0
  };
  // clang-format on

  l.weights_at()[0]->set_data(Tensor<float_t>(weight));
  out        = l.forward({{Tensor<>(in)}});
  out_result = (*out[0]).toTensor()[0];

  for (size_t i = 0; i < out_result.size(); i++) {
    EXPECT_FLOAT_EQ(out_result[i], out_expected[i]);
  }
}

TEST(deconvolutional, fprop_padding_same) {
  deconvolutional_layer l(2, 2, 3, 1, 2, padding::same);
  l.weight_init_f(parameter_init::constant(0.0));

  vec_t in(4, 0);
  vec_t out_expected(8, 0);
  uniform_rand(in.begin(), in.end(), -1.0, 1.0);

  auto out         = l.forward({{Tensor<>(in)}});
  vec_t out_result = (*out[0]).toTensor()[0];

  for (size_t i = 0; i < out_result.size(); i++) {
    EXPECT_FLOAT_EQ(out_result[i], float_t{0});
  }

  // clang-format off
  vec_t weight = {
      0.3,  0.1,  0.2,
      0.0, -0.1, -0.1,
      0.05, -0.2,  0.05,

      0.0, -0.1,  0.1,
      0.1, -0.2,  0.3,
      0.2, -0.3,  0.2
  };

  in = {
      3.0, 2.0,
      3.0, 0.0
  };

  out_expected = {
       0.0,  0.1,
      -0.8, -0.55,

      -0.7,  0.8,
      -1.1,  0.9
  };
  // clang-format on

  // resize tensor because its dimension changed in above used test case
  l.weights_at()[0]->set_data(Tensor<float_t>(weight));
  out        = l.forward({{Tensor<>(in)}});
  out_result = (*out[0]).toTensor()[0];

  for (size_t i = 0; i < out_expected.size(); i++) {
    EXPECT_FLOAT_EQ(out_result[i], out_expected[i]);
  }
}

// TODO(karan): check
/*
TEST(deconvolutional, gradient_check) {  // tanh - mse
  network<sequential> nn;
  nn << deconvolutional_layer(2, 2, 3, 1, 1) << tanh();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second,
                                     epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(deconvolutional, gradient_check2) {  // sigmoid - mse
  network<sequential> nn;
  nn << deconvolutional_layer(2, 2, 3, 1, 1) << sigmoid();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second,
                                     epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(deconvolutional, gradient_check3) {  // rectified - mse
  network<sequential> nn;

  nn << deconvolutional_layer(2, 2, 3, 1, 1) << relu();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second,
                                     epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(deconvolutional, gradient_check4) {  // identity - mse
  network<sequential> nn;

  nn << deconvolutional_layer(2, 2, 3, 1, 1);

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second,
                                     epsilon<float_t>(), GRAD_CHECK_ALL));
}

// TODO: check
TEST(deconvolutional, gradient_check5) {  // sigmoid - cross-entropy
  network<sequential> nn;

  nn << deconvolutional_layer(2, 2, 3, 1, 1) << sigmoid();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<cross_entropy>(
    test_data.first, test_data.second, epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(deconvolutional, read_write) {
  deconvolutional_layer l1(2, 2, 3, 1, 1);
  deconvolutional_layer l2(2, 2, 3, 1, 1);

  l1.init_weight();
  l2.init_weight();

  serialization_test(l1, l2);
}

TEST(deconvolutional, read_write2) {
#define O true
#define X false
  static const bool connection[] = {O, X, X, X, O, O, O, O, X,
                                    X, X, O, O, O, O, X, X, X};
#undef O
#undef X
  deconvolutional_layer layer1(14, 14, 5, 3, 6,
                               connection_table(connection, 3, 6));
  deconvolutional_layer layer2(14, 14, 5, 3, 6,
                               connection_table(connection, 3, 6));
  layer1.init_weight();
  layer2.init_weight();

  serialization_test(layer1, layer2);
}
*/
}  // namespace tiny_dnn
