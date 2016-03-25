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
#include "picotest/picotest.h"
#include "testhelper.h"
#include "tiny_cnn/tiny_cnn.h"

namespace tiny_cnn {

TEST(convolutional, fprop) {
    typedef network<mse, gradient_descent_levenberg_marquardt> CNN;
    CNN nn;

    convolutional_layer<sigmoid> l(5, 5, 3, 1, 2);

    vec_t in(25);

    ASSERT_EQ(l.weight().size(), 18);

    std::fill(l.bias().begin(), l.bias().end(), 0.0);
    std::fill(l.weight().begin(), l.weight().end(), 0.0);

    uniform_rand(in.begin(), in.end(), -1.0, 1.0);

    {
        const vec_t& out = l.forward_propagation(in, 0);

        for (auto o: out)
            EXPECT_DOUBLE_EQ(o, (tiny_cnn::float_t)0.5);

    }

    l.weight()[0] = 0.3;  l.weight()[1] = 0.1; l.weight()[2] = 0.2;
    l.weight()[3] = 0.0;  l.weight()[4] =-0.1; l.weight()[5] =-0.1;
    l.weight()[6] = 0.05; l.weight()[7] =-0.2; l.weight()[8] = 0.05;

    l.weight()[9]  = 0.0; l.weight()[10] =-0.1; l.weight()[11] = 0.1;
    l.weight()[12] = 0.1; l.weight()[13] =-0.2; l.weight()[14] = 0.3;
    l.weight()[15] = 0.2; l.weight()[16] =-0.3; l.weight()[17] = 0.2;

    in[0] = 3;  in[1] = 2;  in[2] = 1;  in[3] = 5; in[4] = 2;
    in[5] = 3;  in[6] = 0;  in[7] = 2;  in[8] = 0; in[9] = 1;
    in[10] = 0; in[11] = 6; in[12] = 1; in[13] = 1; in[14] = 10;
    in[15] = 3; in[16] =-1; in[17] = 2; in[18] = 9; in[19] = 0;
    in[20] = 1; in[21] = 2; in[22] = 1; in[23] = 5; in[24] = 5;

    {
        const vec_t& out = l.forward_propagation(in, 0);

        EXPECT_NEAR(0.4875026, out[0], 1E-5);
        EXPECT_NEAR(0.8388910, out[1], 1E-5);
        EXPECT_NEAR(0.8099984, out[2], 1E-5);
        EXPECT_NEAR(0.7407749, out[3], 1E-5);
        EXPECT_NEAR(0.5000000, out[4], 1E-5);
        EXPECT_NEAR(0.1192029, out[5], 1E-5);
        EXPECT_NEAR(0.5986877, out[6], 1E-5);
        EXPECT_NEAR(0.7595109, out[7], 1E-5);
        EXPECT_NEAR(0.6899745, out[8], 1E-5);
    }


}

TEST(convolutional, bprop) {
    network<cross_entropy, gradient_descent_levenberg_marquardt> nn;

    nn << convolutional_layer<sigmoid>(5, 5, 3, 1, 1);

    vec_t a(25, 0.0), t(9, 0.0);
    std::vector<vec_t> data, train;

    for (int y = 0; y < 5; y++) {
        a[5*y+3] = 1.0;
    }

    for (int y = 0; y < 3; y++) {
        t[3*y+0] = 0.0;
        t[3*y+1] = 0.5;
        t[3*y+2] = 1.0;
    }

    for (int i = 0; i < 100; i++) {
        data.push_back(a);
        train.push_back(t);
    }

    nn.train(data, train);

    vec_t predicted = nn.predict(a);
}

TEST(convolutional, gradient_check) { // tanh - mse
    network<mse, gradient_descent_levenberg_marquardt> nn;
    nn << convolutional_layer<tan_h>(5, 5, 3, 1, 1);

    vec_t a(25, 0.0);
    label_t t = 3;

    uniform_rand(a.begin(), a.end(), -1, 1);
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check(&a, &t, 1, 1e-4, GRAD_CHECK_ALL));
}

TEST(convolutional, gradient_check2) { // sigmoid - mse
    network<mse, gradient_descent_levenberg_marquardt> nn;
    nn << convolutional_layer<sigmoid>(5, 5, 3, 1, 1);

    vec_t a(25, 0.0);
    label_t t = 3;

    uniform_rand(a.begin(), a.end(), -1, 1);
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check(&a, &t, 1, 1e-4, GRAD_CHECK_ALL));
}

TEST(convolutional, gradient_check3) { // rectified - mse
    network<mse, gradient_descent_levenberg_marquardt> nn;

    nn << convolutional_layer<rectified_linear>(5, 5, 3, 1, 1);

    vec_t a(25, 0.0);
    label_t t = 3;

    uniform_rand(a.begin(), a.end(), -1, 1);
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check(&a, &t, 1, 1e-4, GRAD_CHECK_ALL));
}

TEST(convolutional, gradient_check4) { // identity - mse
    network<mse, gradient_descent_levenberg_marquardt> nn;

    nn << convolutional_layer<identity>(5, 5, 3, 1, 1);

    vec_t a(25, 0.0);
    label_t t = 3;

    uniform_rand(a.begin(), a.end(), -1, 1);
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check(&a, &t, 1, 1e-4, GRAD_CHECK_ALL));
}

TEST(convolutional, gradient_check5) { // sigmoid - cross-entropy
    network<cross_entropy, gradient_descent_levenberg_marquardt> nn;

    nn << convolutional_layer<sigmoid>(5, 5, 3, 1, 1);

    vec_t a(25, 0.0);
    label_t t = 3;

    uniform_rand(a.begin(), a.end(), -1, 1);
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check(&a, &t, 1, 1e-4, GRAD_CHECK_ALL));
}

TEST(convolutional, read_write)
{
    convolutional_layer<tan_h> l1(5, 5, 3, 1, 1);
    convolutional_layer<tan_h> l2(5, 5, 3, 1, 1);

    l1.init_weight();
    l2.init_weight();

    serialization_test(l1, l2);
}

TEST(convolutional, read_write2) {
#define O true
#define X false
    static const bool connection[] = {
        O, X, X, X, O, O,
        O, O, X, X, X, O,
        O, O, O, X, X, X
    };
#undef O
#undef X
    convolutional_layer<tan_h> layer1(14, 14, 5, 3, 6, connection_table(connection, 3, 6));
    convolutional_layer<tan_h> layer2(14, 14, 5, 3, 6, connection_table(connection, 3, 6));
    layer1.init_weight();
    layer2.init_weight();

    serialization_test(layer1, layer2);
}


} // namespace tiny-cnn
