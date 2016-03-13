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

TEST(multi_layer, gradient_check) { // sigmoid - cross-entropy
    typedef cross_entropy loss_func;
    typedef sigmoid activation;
    typedef network<loss_func, gradient_descent_levenberg_marquardt> network;

    network nn;
    nn << fully_connected_layer<activation>(10, 14*14*3)
       << convolutional_layer<activation>(14, 14, 5, 3, 6)
       << average_pooling_layer<activation>(10, 10, 6, 2)
       << fully_connected_layer<activation>(5*5*6, 3);

    vec_t a(10, 0.0);
    label_t t = 2;

    uniform_rand(a.begin(), a.end(), -1, 1);
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check(&a, &t, 1, 1e-4, GRAD_CHECK_RANDOM));
}

TEST(multi_layer, gradient_check2) { // tan_h - mse
    typedef mse loss_func;
    typedef tan_h activation;
    typedef network<loss_func, gradient_descent_levenberg_marquardt> network;

    network nn;
    nn << fully_connected_layer<activation>(10, 14 * 14 * 3)
        << convolutional_layer<activation>(14, 14, 5, 3, 6)
        << average_pooling_layer<activation>(10, 10, 6, 2)
        << fully_connected_layer<activation>(5 * 5 * 6, 3);

    vec_t a(10, 0.0);
    label_t t = 2;

    uniform_rand(a.begin(), a.end(), -1, 1);
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check(&a, &t, 1, 1e-4, GRAD_CHECK_RANDOM));
}

TEST(multi_layer, gradient_check3) { // mixture - mse
    typedef mse loss_func;
    typedef network<loss_func, gradient_descent_levenberg_marquardt> network;

    network nn;
    nn << fully_connected_layer<tan_h>(10, 14 * 14 * 3)
        << convolutional_layer<sigmoid>(14, 14, 5, 3, 6)
        << average_pooling_layer<rectified_linear>(10, 10, 6, 2)
        << fully_connected_layer<identity>(5 * 5 * 6, 3);

    vec_t a(10, 0.0);
    label_t t = 2;

    uniform_rand(a.begin(), a.end(), -1, 1);
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check(&a, &t, 1, 1e-4, GRAD_CHECK_RANDOM));
}

TEST(multi_layer4, gradient_check) { // sigmoid - cross-entropy
    typedef cross_entropy loss_func;
    typedef sigmoid activation;
    typedef network<loss_func, gradient_descent_levenberg_marquardt> network;

    network nn;
    nn << fully_connected_layer<activation>(10, 14 * 14 * 3)
        << convolutional_layer<activation>(14, 14, 5, 3, 6)
        << average_pooling_layer<activation>(10, 10, 6, 2)
        << fully_connected_layer<activation>(5 * 5 * 6, 3);

    vec_t a(10, 0.0);
    label_t t = 2;

    uniform_rand(a.begin(), a.end(), -1, 1);
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check(&a, &t, 1, 1e-4, GRAD_CHECK_RANDOM));
}

TEST(multi_layer5, gradient_check) { // softmax - cross-entropy
    typedef cross_entropy loss_func;
    typedef softmax activation;
    typedef network<loss_func, gradient_descent_levenberg_marquardt> network;

    network nn;
    nn << fully_connected_layer<activation>(10, 14 * 14 * 3)
        << convolutional_layer<activation>(14, 14, 5, 3, 6)
        << average_pooling_layer<activation>(10, 10, 6, 2)
        << fully_connected_layer<activation>(5 * 5 * 6, 3);

    vec_t a(10, 0.0);
    label_t t = 2;

    uniform_rand(a.begin(), a.end(), -1, 1);
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check(&a, &t, 1, 5e-3, GRAD_CHECK_RANDOM));
}

TEST(multi_layer6, gradient_check) { // sigmoid - cross-entropy
    typedef cross_entropy loss_func;
    typedef sigmoid activation;
    typedef network<loss_func, gradient_descent_levenberg_marquardt> network;

    network nn;
    nn << fully_connected_layer<activation>(3, 201)
        << fully_connected_layer<activation>(201, 2);

    vec_t a(3, 0.0);
    label_t t = 1;

    uniform_rand(a.begin(), a.end(), 0, 3);
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check(&a, &t, 1, 1e-4, GRAD_CHECK_ALL));
}

TEST(read_write, network)
{
    typedef mse loss_func;
    typedef network<loss_func, gradient_descent_levenberg_marquardt> network;

    network n1, n2;

    n1 << convolutional_layer<tan_h>(32, 32, 5, 1, 6) // C1, 1@32x32-in, 6@28x28-out
        << average_pooling_layer<tan_h>(28, 28, 6, 2) // S2, 6@28x28-in, 6@14x14-out
        << convolutional_layer<tan_h>(14, 14, 5, 6, 16) // C3, 6@14x14-in, 16@10x10-in
        << average_pooling_layer<tan_h>(10, 10, 16, 2) // S4, 16@10x10-in, 16@5x5-out
        << convolutional_layer<tan_h>(5, 5, 5, 16, 120) // C5, 16@5x5-in, 120@1x1-out
        << fully_connected_layer<tan_h>(120, 10); // F6, 120-in, 10-out

    n2 << convolutional_layer<tan_h>(32, 32, 5, 1, 6) // C1, 1@32x32-in, 6@28x28-out
        << average_pooling_layer<tan_h>(28, 28, 6, 2) // S2, 6@28x28-in, 6@14x14-out
        << convolutional_layer<tan_h>(14, 14, 5, 6, 16) // C3, 6@14x14-in, 16@10x10-in
        << average_pooling_layer<tan_h>(10, 10, 16, 2) // S4, 16@10x10-in, 16@5x5-out
        << convolutional_layer<tan_h>(5, 5, 5, 16, 120) // C5, 16@5x5-in, 120@1x1-out
        << fully_connected_layer<tan_h>(120, 10); // F6, 120-in, 10-out

    n1.init_weight();
    n2.init_weight();

    std::vector<vec_t> t;
    std::vector<label_t> l;
    t.push_back(vec_t(32*32, 0.0));
    l.push_back(3);
    n1.train(t, l, 1, 1);

    serialization_test(n1, n2);

    vec_t in(32*32, 0.0);

    auto res1 = n1.predict(in);
    auto res2 = n2.predict(in);

    ASSERT_TRUE(n1.has_same_weights(n2, 1e-6));

    for (int i = 0; i < 10; i++) {
        tiny_cnn::float_t eps = std::abs(res1[i]) * 1e-5;
        ASSERT_TRUE(std::abs(res1[i] - res2[i]) < eps);
    }
}

} // namespace tiny-cnn
