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

TEST(fully_connected, train) {
    network<sequential> nn;
    adagrad optimizer;

    nn << fully_connected_layer<sigmoid>(3, 2);

    vec_t a(3), t(2), a2(3), t2(2);

    a[0] = 3.0f; a[1] = 0.0f; a[2] = -1.0f;
    t[0] = 0.3f; t[1] = 0.7f;

    a2[0] = 0.2f; a2[1] = 0.5f; a2[2] = 4.0f;
    t2[0] = 0.5f; t2[1] = 0.1f;

    std::vector<vec_t> data, train;

    for (int i = 0; i < 100; i++) {
        data.push_back(a);
        data.push_back(a2);
        train.push_back(t);
        train.push_back(t2);
    }
    optimizer.alpha = 0.1f;
    nn.train<mse>(optimizer, data, train, 1, 10);

    vec_t predicted = nn.predict(a);

    EXPECT_NEAR(predicted[0], t[0], 1E-5);
    EXPECT_NEAR(predicted[1], t[1], 1E-5);

    predicted = nn.predict(a2);

    EXPECT_NEAR(predicted[0], t2[0], 1E-5);
    EXPECT_NEAR(predicted[1], t2[1], 1E-5);
}

TEST(fully_connected, train2) {
    network<sequential> nn;
    gradient_descent optimizer;

    nn << fully_connected_layer<tan_h>(4, 6)
       << fully_connected_layer<tan_h>(6, 3);

    vec_t a(4, 0.0), t(3, 0.0), a2(4, 0.0), t2(3, 0.0);

    a[0] = 3.0f; a[1] = 1.0f; a[2] = -1.0f; a[3] = 4.0f;
    t[0] = 0.3f; t[1] = 0.7f; t[2] = 0.3f;

    a2[0] = 1.0f; a2[1] = 0.0f; a2[2] = 4.0f; a2[3] = 2.0f;
    t2[0] = 0.6f; t2[1] = 0.0f; t2[2] = 0.1f;

    std::vector<vec_t> data, train;

    for (int i = 0; i < 100; i++) {
        data.push_back(a);
        data.push_back(a2);
        train.push_back(t);
        train.push_back(t2);
    }
    optimizer.alpha = 0.1f;
    nn.train<mse>(optimizer, data, train, 1, 10);

    vec_t predicted = nn.predict(a);

    EXPECT_NEAR(predicted[0], t[0], 1E-4);
    EXPECT_NEAR(predicted[1], t[1], 1E-4);

    predicted = nn.predict(a2);

    EXPECT_NEAR(predicted[0], t2[0], 1E-4);
    EXPECT_NEAR(predicted[1], t2[1], 1E-4);
}

TEST(fully_connected, gradient_check) {
    network<sequential> nn;
    nn << fully_connected_layer<tan_h>(50, 10);

    const auto test_data = generate_gradient_check_data(nn.in_data_size());
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second, epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(fully_connected, read_write)
{
    fully_connected_layer<tan_h> l1(100, 100);
    fully_connected_layer<tan_h> l2(100, 100);

    l1.setup(true);
    l2.setup(true);

    serialization_test(l1, l2);
}

TEST(fully_connected, forward)
{
    fully_connected_layer<identity> l(4, 2);
    EXPECT_EQ(l.in_channels(), serial_size_t(3)); // in, W and b

    l.weight_init(weight_init::constant(1.0));
    l.bias_init(weight_init::constant(0.5));

    vec_t in = {0,1,2,3};
    vec_t out = l.forward({ {in} })[0][0];
    vec_t out_expected = {6.5, 6.5}; // 0+1+2+3+0.5

    for (size_t i = 0; i < out_expected.size(); i++) {
        EXPECT_FLOAT_EQ(out_expected[i], out[i]);
    }
}

#ifdef CNN_USE_NNPACK
TEST(fully_connected, forward_nnp)
{
    nnp_initialize();
    fully_connected_layer<identity> l(4, 2, true, core::backend_t::nnpack);
    EXPECT_EQ(l.in_channels(), 3); // in, W and b

    l.weight_init(weight_init::constant(1.0));
    l.bias_init(weight_init::constant(0.5));

    vec_t in = {0,1,2,3};
    vec_t out = l.forward({ {in} })[0][0];
    vec_t out_expected = {6.5, 6.5}; // 0+1+2+3+0.5

    for (size_t i = 0; i < out_expected.size(); i++) {
        EXPECT_FLOAT_EQ(out_expected[i], out[i]);
    }
}
#endif

TEST(fully_connected, forward_nobias)
{
    fully_connected_layer<identity> l(4, 2, false);
    EXPECT_EQ(l.in_channels(), serial_size_t(2));// in and W

    l.weight_init(weight_init::constant(1.0));

    vec_t in = { 0,1,2,3 };
    vec_t out = l.forward({ { in } })[0][0];
    vec_t out_expected = { 6.0, 6.0 }; // 0+1+2+3

    for (size_t i = 0; i < out_expected.size(); i++) {
        EXPECT_FLOAT_EQ(out_expected[i], out[i]);
    }
}

} // namespace tiny-dnn
