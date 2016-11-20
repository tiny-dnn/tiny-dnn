/*
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
 #include "gtest/gtest.h"
#include "testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

namespace tiny_dnn {

/*
TEST(quantized_fully_connected, train) {
    network<sequential> nn;
    adagrad optimizer;

    nn << quantized_fully_connected_layer<sigmoid>(3, 2);

    vec_t a(3), t(2), a2(3), t2(2);

    a[0] = 3.0; a[1] = 0.0; a[2] = -1.0;
    t[0] = 0.3; t[1] = 0.7;

    a2[0] = 0.2; a2[1] = 0.5; a2[2] = 4.0;
    t2[0] = 0.5; t2[1] = 0.1;

    std::vector<vec_t> data, train;

    for (int i = 0; i < 100; i++) {
        data.push_back(a);
        data.push_back(a2);
        train.push_back(t);
        train.push_back(t2);
    }
    optimizer.alpha = 0.1;
    nn.train<mse>(optimizer, data, train, 1, 10);

    vec_t predicted = nn.predict(a);

    EXPECT_NEAR(predicted[0], t[0], 2E-2);
    EXPECT_NEAR(predicted[1], t[1], 2E-2);

    predicted = nn.predict(a2);

    EXPECT_NEAR(predicted[0], t2[0], 5E-2);
    EXPECT_NEAR(predicted[1], t2[1], 5E-2);
}

TEST(quantized_fully_connected, train2) {
    network<sequential> nn;
    gradient_descent optimizer;

    nn << quantized_fully_connected_layer<tan_h>(4, 6)
       << quantized_fully_connected_layer<tan_h>(6, 3);

    vec_t a(4, 0.0), t(3, 0.0), a2(4, 0.0), t2(3, 0.0);

    a[0] = 3.0; a[1] = 1.0; a[2] = -1.0; a[3] = 4.0;
    t[0] = 0.3; t[1] = 0.7; t[2] = 0.3;

    a2[0] = 1.0; a2[1] = 0.0; a2[2] = 4.0; a2[3] = 2.0;
    t2[0] = 0.6; t2[1] = 0.0; t2[2] = 0.1;

    std::vector<vec_t> data, train;

    for (int i = 0; i < 100; i++) {
        data.push_back(a);
        data.push_back(a2);
        train.push_back(t);
        train.push_back(t2);
    }
    optimizer.alpha = 0.1;
    nn.train<mse>(optimizer, data, train, 1, 10);

    vec_t predicted = nn.predict(a);

    EXPECT_NEAR(predicted[0], t[0], 5E-2);
    EXPECT_NEAR(predicted[1], t[1], 5E-2);

    predicted = nn.predict(a2);

    EXPECT_NEAR(predicted[0], t2[0], 5E-2);
    EXPECT_NEAR(predicted[1], t2[1], 5E-2);
}

TEST(quantized_fully_connected, gradient_check) {
    network<sequential> nn;
    nn << quantized_fully_connected_layer<tan_h>(50, 10);

    vec_t a(50, 0.0);
    label_t t = 9;

    uniform_rand(a.begin(), a.end(), -1, 1);
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check<mse>(&a, &t, 1, epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(quantized_fully_connected, read_write)
{
    quantized_fully_connected_layer<tan_h> l1(100, 100);
    quantized_fully_connected_layer<tan_h> l2(100, 100);

    l1.setup(true);
    l2.setup(true);

    quantized_serialization_test(l1, l2);
}*/

TEST(quantized_fully_connected, forward)
{
    quantized_fully_connected_layer<identity> l(4, 2);
    EXPECT_EQ(l.in_channels(), serial_size_t(3)); // in, W and b

    l.weight_init(weight_init::constant(1.0));
    l.bias_init(weight_init::constant(0.5));

    vec_t in = {0,1,2,3};
    vec_t out = l.forward({ { in } })[0][0];
    vec_t out_expected = {6.5, 6.5}; // 0+1+2+3+0.5

    for (size_t i = 0; i < out_expected.size(); i++) {
        EXPECT_NEAR(out_expected[i], out[i], 1E-2);
    }
}

#ifdef CNN_USE_NNPACK
TEST(quantized_fully_connected, forward_nnp)
{
    quantized_fully_connected_layer<identity> l(4, 2, true, core::backend_t::nnpack);
    EXPECT_EQ(l.in_channels(), serial_size_t(3)); // in, W and b

    l.weight_init(weight_init::constant(1.0));
    l.bias_init(weight_init::constant(0.5));

    vec_t in = {0,1,2,3};
    vec_t out = l.forward({ {in} })[0][0];
    vec_t out_expected = {6.5, 6.5}; // 0+1+2+3+0.5

    for (size_t i = 0; i < out_expected.size(); i++) {
        EXPECT_NEAR(out_expected[i], out[i], 1E-2);
    }
}
#endif

TEST(quantized_fully_connected, forward_nobias)
{
    quantized_fully_connected_layer<identity> l(4, 2, false);
    EXPECT_EQ(l.in_channels(), serial_size_t(2));// in and W

    l.weight_init(weight_init::constant(1.0));

    vec_t in = { 0,1,2,3 };
    vec_t out = l.forward({ { in } })[0][0];
    vec_t out_expected = { 6.0, 6.0 }; // 0+1+2+3

    for (size_t i = 0; i < out_expected.size(); i++) {
        EXPECT_NEAR(out_expected[i], out[i], 1E-2);
    }
}

} // namespace tiny-dnn
