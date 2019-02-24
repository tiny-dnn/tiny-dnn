/*
    Copyright (c) 2019, Dejan Milosavljevic and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <vector>

namespace tiny_dnn {


//! By the book forward. Any kind of optimizations are not welcome here.
float_t softclip_forward_function( float_t const& x, float_t const& alpha ){
    return  ( float_t(1)/alpha)* log( ( float_t(1)+ exp( alpha *x ) )/( float_t(1) + exp(alpha*(x-float_t(1)))) );
}

TEST(softclip, forward) {
  float_t alpha = 2.0;
  softclip_layer sc(shape3d(10, 1, 1), alpha );

  tensor_t in = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
    { -9, -8, -7, -6, -5, -4, -3, -2, -1, 0},
    {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f},
    {-0.0f, -0.1f, -0.2f, -0.3f, -0.4f, -0.5f, -0.6f, -0.7f, -0.8f, -0.9f },
  };

   tensor_t out_expected;
   out_expected.resize( in. size() );
   for( int index=0; index < in.size(); ++index ) {
     auto const &row = in[index];
       for( auto & value: row ) {
         out_expected[index].push_back( softclip_forward_function( value, alpha ) );
       }
   }

  std::vector<const tensor_t*> out;
  sc.forward( {in}, out );

  for (size_t i = 0; i < 10; i++) {
    ASSERT_NEAR(out_expected[0][i], (*out[0])[0][i], sc.epsilon_value () );
    ASSERT_NEAR(out_expected[1][i], (*out[0])[1][i], sc.epsilon_value () );
    ASSERT_NEAR(out_expected[2][i], (*out[0])[2][i], sc.epsilon_value () );
    ASSERT_NEAR(out_expected[3][i], (*out[0])[3][i], sc.epsilon_value () );
  }
}

TEST(softclip, gradient_check) {
  const size_t width    = 2;
  const size_t height   = 2;
  const size_t channels = 10;
  softclip_layer sc(shape3d(height, width, channels), 2.0 );
  std::vector<tensor_t> input_data =
    generate_test_data({1}, {width * height * channels});
  std::vector<tensor_t> in_grad = input_data;  // copy constructor
  std::vector<tensor_t> out_data =
    generate_test_data({1}, {width * height * channels});
  std::vector<tensor_t> out_grad =
    generate_test_data({1}, {width * height * channels});
  const size_t trials = 100;
  for (size_t i = 0; i < trials; i++) {
    const size_t in_edge  = uniform_idx(input_data);
    const size_t in_idx   = uniform_idx(input_data[in_edge][0]);
    const size_t out_edge = uniform_idx(out_data);
    const size_t out_idx  = uniform_idx(out_data[out_edge][0]);
    float_t ngrad = numeric_gradient(sc, input_data, in_edge, in_idx, out_data,
                                     out_edge, out_idx);
    float_t cgrad = analytical_gradient(sc, input_data, in_edge, in_idx,
                                        out_data, out_grad, out_edge, out_idx);
    EXPECT_NEAR(ngrad, cgrad, epsilon<float_t>());
  }
}

}  // namespace tiny_dnn
