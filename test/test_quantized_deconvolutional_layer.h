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

using namespace tiny_dnn::activation;

namespace tiny_dnn {

TEST(quantized_deconvolutional, setup_internal) {
  quantized_deconvolutional_layer l(2, 2, 3, 1, 2, padding::valid, true, 1, 1,
                                    backend_t::internal);

  EXPECT_EQ(l.parallelize(), true);              // if layer can be parallelized
  EXPECT_EQ(l.in_channels(), serial_size_t(3));  // num of input tensors
  EXPECT_EQ(l.out_channels(), serial_size_t(1));    // num of output tensors
  EXPECT_EQ(l.in_data_size(), serial_size_t(4));    // size of input tensors
  EXPECT_EQ(l.out_data_size(), serial_size_t(32));  // size of output tensors
  EXPECT_EQ(l.in_data_shape().size(),
            serial_size_t(1));  // number of inputs shapes
  EXPECT_EQ(l.out_data_shape().size(),
            serial_size_t(1));                      // num of output shapes
  EXPECT_EQ(l.weights().size(), serial_size_t(2));  // the wieghts vector size
  EXPECT_EQ(l.weights_grads().size(),
            serial_size_t(2));                       // the wieghts vector size
  EXPECT_EQ(l.inputs().size(), serial_size_t(3));    // num of input edges
  EXPECT_EQ(l.outputs().size(), serial_size_t(1));   // num of outpus edges
  EXPECT_EQ(l.in_types().size(), serial_size_t(3));  // num of input data types
  EXPECT_EQ(l.out_types().size(),
            serial_size_t(1));                   // num of output data types
  EXPECT_EQ(l.fan_in_size(), serial_size_t(9));  // num of incoming connections
  EXPECT_EQ(l.fan_out_size(),
            serial_size_t(18));  // num of outgoing connections
  EXPECT_STREQ(l.layer_type().c_str(), "q_deconv");  // string with layer type
}

TEST(quantized_deconvolutional, fprop) {
  quantized_deconvolutional_layer l(2, 2, 3, 1, 2);

  // layer::forward_propagation expects tensors, even if we feed only one
  // input
  // at a time
  auto create_simple_tensor = [](size_t vector_size) {
    return tensor_t(1, vec_t(vector_size));
  };

  // create simple tensors that wrap the payload vectors of the correct size
  tensor_t in_tensor     = create_simple_tensor(4),
           out_tensor    = create_simple_tensor(32),
           weight_tensor = create_simple_tensor(18),
           bias_tensor   = create_simple_tensor(2);

  // short-hand references to the payload vectors
  vec_t &in = in_tensor[0], &out = out_tensor[0], &weight = weight_tensor[0];

  ASSERT_EQ(l.in_shape()[1].size(), serial_size_t(18));  // weight

  uniform_rand(in.begin(), in.end(), -1.0, 1.0);

  std::vector<tensor_t *> in_data, out_data;
  in_data.push_back(&in_tensor);
  in_data.push_back(&weight_tensor);
  in_data.push_back(&bias_tensor);
  out_data.push_back(&out_tensor);
  l.setup(false);
  {
    l.forward_propagation(in_data, out_data);

    for (auto o : out) EXPECT_NEAR(0.0, o, 1E-3);
  }

  // clang-format off
    weight[0] = 0.3f;  weight[1] = 0.1f; weight[2] = 0.2f;
    weight[3] = 0.0f;  weight[4] =-0.1f; weight[5] =-0.1f;
    weight[6] = 0.05f; weight[7] =-0.2f; weight[8] = 0.05f;

    weight[9]  = 0.0f; weight[10] =-0.1f; weight[11] = 0.1f;
    weight[12] = 0.1f; weight[13] =-0.2f; weight[14] = 0.3f;
    weight[15] = 0.2f; weight[16] =-0.3f; weight[17] = 0.2f;

    in[0] = 3;  in[1] = 2;
    in[2] = 3;  in[3] = 0;

    {
        l.forward_propagation(in_data, out_data);

        EXPECT_NEAR(0.9017647, out[0], 1E-2);
        EXPECT_NEAR(0.9017647, out[1], 1E-2);
        EXPECT_NEAR(0.8049019, out[2], 1E-2);
        EXPECT_NEAR(0.4025490, out[3], 1E-2);
        EXPECT_NEAR(0.9017647, out[4], 1E-2);
        EXPECT_NEAR(0.0001960, out[5], 1E-2);
        EXPECT_NEAR(0.1045097, out[6], 1E-2);
        EXPECT_NEAR(-0.200980, out[7], 1E-2);
        EXPECT_NEAR(0.1566666, out[8], 1E-2);
        EXPECT_NEAR(-0.797058, out[9], 1E-2);
        EXPECT_NEAR(-0.551176, out[10], 1E-2);
        EXPECT_NEAR(0.1045097, out[11], 1E-2);
        EXPECT_NEAR(0.1566666, out[12], 1E-2);
        EXPECT_NEAR(-0.603333, out[13], 1E-2);
        EXPECT_NEAR(0.1566666, out[14], 1E-2);
        EXPECT_NEAR(0.0001960, out[15], 1E-2);
    }
  // clang-format on
}

TEST(quantized_deconvolutional, fprop2) {
  quantized_deconvolutional_layer l(2, 2, 3, 1, 2, padding::same);

  // layer::forward_propagation expects tensors, even if we feed only one
  // input at a time
  auto create_simple_tensor = [](size_t vector_size) {
    return tensor_t(1, vec_t(vector_size));
  };

  // create simple tensors that wrap the payload vectors of the correct size
  tensor_t in_tensor     = create_simple_tensor(4),
           out_tensor    = create_simple_tensor(32),
           weight_tensor = create_simple_tensor(18),
           bias_tensor   = create_simple_tensor(2);

  // short-hand references to the payload vectors
  vec_t &in = in_tensor[0], &out = out_tensor[0], &weight = weight_tensor[0];

  ASSERT_EQ(l.in_shape()[1].size(), serial_size_t(18));  // weight

  uniform_rand(in.begin(), in.end(), -1.0, 1.0);

  std::vector<tensor_t *> in_data, out_data;
  in_data.push_back(&in_tensor);
  in_data.push_back(&weight_tensor);
  in_data.push_back(&bias_tensor);
  out_data.push_back(&out_tensor);

  l.setup(false);
  {
    l.forward_propagation(in_data, out_data);

    for (auto o : out) EXPECT_NEAR(0.0, o, 1E-3);
  }

  // clang-format off
  weight[0] = 0.3f;  weight[1] = 0.1f; weight[2] = 0.2f;
  weight[3] = 0.0f;  weight[4] =-0.1f; weight[5] =-0.1f;
  weight[6] = 0.05f; weight[7] =-0.2f; weight[8] = 0.05f;

  weight[9]  = 0.0f; weight[10] =-0.1f; weight[11] = 0.1f;
  weight[12] = 0.1f; weight[13] =-0.2f; weight[14] = 0.3f;
  weight[15] = 0.2f; weight[16] =-0.3f; weight[17] = 0.2f;

  in[0] = 3;  in[1] = 2;
  in[2] = 3;  in[3] = 0;

  {
    l.forward_propagation(in_data, out_data);

    EXPECT_NEAR(0.0001960, out[0], 1E-2);
    EXPECT_NEAR(0.1045097, out[1], 1E-2);
    EXPECT_NEAR(-0.797058, out[2], 1E-2);
    EXPECT_NEAR(-0.551176, out[3], 1E-2);
    EXPECT_NEAR(-0.707647, out[4], 1E-2);
    EXPECT_NEAR(0.8049019, out[5], 1E-2);
    EXPECT_NEAR(-0.797058, out[6], 1E-2);
    EXPECT_NEAR(1.1029412, out[7], 1E-2);
    EXPECT_NEAR(0.1566666, out[8], 1E-2);
    EXPECT_NEAR(-0.797058, out[9], 1E-2);
    EXPECT_NEAR(-0.551176, out[10], 1E-2);
    EXPECT_NEAR(0.1045097, out[11], 1E-2);
    EXPECT_NEAR(0.1566666, out[12], 1E-2);
    EXPECT_NEAR(-0.603333, out[13], 1E-2);
    EXPECT_NEAR(0.1566666, out[14], 1E-2);
    EXPECT_NEAR(0.0001960, out[15], 1E-2);
  }
  // clang-format on
}

/*
TEST(quantized_deconvolutional, gradient_check) { // tanh - mse
    network<sequential> nn;
    nn << quantized_deconvolutional_layer(5, 5, 3, 1, 1) << tanh();

    vec_t a(25, 0.0);
    label_t t = 3;

    uniform_rand(a.begin(), a.end(), -1, 1);
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check<mse>(&a, &t, 1, epsilon<float_t>(),
GRAD_CHECK_ALL));
}

TEST(quantized_deconvolutional, gradient_check2) { // sigmoid - mse
    network<sequential> nn;
    nn << quantized_deconvolutional_layer(5, 5, 3, 1, 1) << sigmoid();

    vec_t a(25, 0.0);
    label_t t = 3;

    uniform_rand(a.begin(), a.end(), -1, 1);
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check<mse>(&a, &t, 1, epsilon<float_t>(),
GRAD_CHECK_ALL));
}

TEST(quantized_deconvolutional, gradient_check3) { // rectified - mse
    network<sequential> nn;
    nn << quantized_deconvolutional_layer(5, 5, 3, 1, 1) << tanh();

    vec_t a(25, 0.0);
    label_t t = 3;

    uniform_rand(a.begin(), a.end(), -1, 1);
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check<mse>(&a, &t, 1, epsilon<float_t>(),
GRAD_CHECK_ALL));
}

TEST(quantized_deconvolutional, gradient_check4) { // identity - mse
    network<sequential> nn;
    nn << quantized_deconvolutional_layer(5, 5, 3, 1, 1);

    vec_t a(25, 0.0);
    label_t t = 3;

    uniform_rand(a.begin(), a.end(), -1, 1);
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check<mse>(&a, &t, 1, epsilon<float_t>(),
GRAD_CHECK_ALL));
}

TEST(quantized_deconvolutional, gradient_check5) { // sigmoid - cross-entropy
    network<sequential> nn;
    nn << quantized_deconvolutional_layer(5, 5, 3, 1, 1) << sigmoid();

    vec_t a(25, 0.0);
    label_t t = 3;

    uniform_rand(a.begin(), a.end(), -1, 1);
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check<cross_entropy>(&a, &t, 1, epsilon<float_t>(),
GRAD_CHECK_ALL));
}

TEST(quantized_deconvolutional, read_write)
{
    quantized_deconvolutional_layer l1(5, 5, 3, 1, 1);
    quantized_deconvolutional_layer l2(5, 5, 3, 1, 1);

    l1.init_weight();
    l2.init_weight();

    serialization_test(l1, l2);
}

TEST(quantized_deconvolutional, read_write2) {
#define O true
#define X false
    static const bool connection[] = {
        O, X, X, X, O, O,
        O, O, X, X, X, O,
        O, O, O, X, X, X
    };
#undef O
#undef X
    quantized_deconvolutional_layer layer1(14, 14, 5, 3, 6,
connection_table(connection, 3, 6));
    quantized_deconvolutional_layer layer2(14, 14, 5, 3, 6,
connection_table(connection, 3, 6));
    layer1.init_weight();
    layer2.init_weight();

    serialization_test(layer1, layer2);
}*/

}  // namespace tiny-dnn
