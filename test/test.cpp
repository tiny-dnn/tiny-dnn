/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#define CATCH_CONFIG_FAST_COMPILE
#include "third_party/catch2/catch.hpp"

#define STR(x) #x 
#define CONCAT(x,y) STR(x ## _ ## y) 
#define TEST(x,y) TEST_CASE(CONCAT(x,y)) 

#define ASSERT_TRUE(x) REQUIRE(x) 
#define ASSERT_EQ(x,y) REQUIRE( (x) == (y) ) 
#define ASSERT_NEAR(x,y,tolerance) REQUIRE( (x) == Approx(y).margin(tolerance) ) 
#define ASSERT_DOUBLE_EQ(x,y) REQUIRE( (x) == Approx(y) ) 
#define ASSERT_FLOAT_EQ(x,y) REQUIRE( (x) == Approx(y) ) 

#define EXPECT_EQ(x,y) CHECK( (x) == (y) ) 
#define EXPECT_STREQ(x,y) CHECK( strcmp((x),(y)) == 0 ) 
#define EXPECT_NE(x,y) CHECK( (x) != (y) ) 
#define EXPECT_DOUBLE_EQ(x,y) CHECK( (x) == Approx(y) ) 
#define EXPECT_FLOAT_EQ(x,y) CHECK( (x) == Approx(y) ) 
#define EXPECT_TRUE(x) CHECK(x) 
#define EXPECT_FALSE(x) CHECK(!(x)) 
#define EXPECT_NEAR(x,y,tolerance) CHECK( (x) == Approx(y).margin(tolerance) ) 
#define EXPECT_LT(x,y) CHECK((x) < (y))
#define EXPECT_LE(x,y) CHECK((x) <= (y))
#define EXPECT_GE(x,y) CHECK((x) >= (y))
#define EXPECT_THROW(stmt, exc_type) CHECK_THROWS_AS( stmt, exc_type )

#include "tiny_dnn/tiny_dnn.h"
#include "test/testhelper.h"

using namespace tiny_dnn::activation;

#include "test_activation_layer.h"
#include "test_average_pooling_layer.h"
// TODO(yida): fix broken test
// #include "test_average_unpooling_layer.h"
#include "test_batch_norm_layer.h"
#include "test_concat_layer.h"
#include "test_convolutional_layer.h"
#include "test_core.h"
#include "test_deconvolutional_layer.h"
#include "test_dropout_layer.h"
#include "test_fully_connected_layer.h"
#include "test_global_average_pooling_layer.h"
#include "test_integration.h"
#include "test_l2_norm_layer.h"
#include "test_large_thread_count.h"
#include "test_lrn_layer.h"
#include "test_max_pooling_layer.h"
#include "test_models.h"
#include "test_network.h"
#include "test_node.h"
#include "test_nodes.h"
#include "test_optimizers.h"
#include "test_power_layer.h"
#include "test_quantization.h"
#include "test_quantized_convolutional_layer.h"
#include "test_quantized_deconvolutional_layer.h"
#include "test_slice_layer.h"
#include "test_target_cost.h"
#include "test_tensor.h"
#include "test_zero_pad_layer.h"

#include "test_gru_cell.h"
#include "test_lstm_cell.h"
#include "test_rnn_cell.h"

#include "test_nms.h"

#ifndef CNN_NO_SERIALIZATION
#include "test_serialization.h"
#endif  // CNN_NO_SERIALIZATION

#ifdef CNN_USE_GEMMLOWP
#include "test_quantized_fully_connected_layer.h"
#endif  // CNN_USE_GEMMLOOP

#ifdef CNN_USE_CAFFE_CONVERTER
#include "test_caffe_converter.h"
#endif  // CNN_USE_CAFFE_CONVERTER

#ifdef DNN_USE_IMAGE_API
#include "test_image.h"
#endif  // DNN_USE_IMAGE_API

