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
#include "picotest.h"
#include "tiny_cnn/tiny_cnn.h"
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

using namespace tiny_cnn;
using namespace tiny_cnn::activation;

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

TEST(convolutional, serialize) {
    convolutional_layer<tan_h> layer1(14, 14, 5, 1, 2);
    convolutional_layer<tan_h> layer2(14, 14, 5, 1, 2);

    vec_t v(14*14);

    uniform_rand(v.begin(), v.end(), -1.0, 1.0);
    layer1.init_weight();

    std::ostringstream os;
    layer1.save(os);

    std::istringstream is(os.str());
    layer2.load(is);

    const vec_t& out1 = layer1.forward_propagation(v, 0);
    const vec_t& out2 = layer2.forward_propagation(v, 0);

    for (size_t i = 0; i < out1.size(); i++)
        EXPECT_NEAR(out1[i], out2[i], 1e-4);
}

TEST(convolutional, serialize2) {
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

    vec_t v(14*14*3);

    uniform_rand(v.begin(), v.end(), -1.0, 1.0);
    layer1.init_weight();

    std::ostringstream os;
    layer1.save(os);

    std::istringstream is(os.str());
    layer2.load(is);

    const vec_t& out1 = layer1.forward_propagation(v, 0);
    const vec_t& out2 = layer2.forward_propagation(v, 0);

    for (size_t i = 0; i < out1.size(); i++)
        EXPECT_NEAR(out1[i], out2[i], 1e-4);
}

TEST(fully_connected, bprop) {
    network<cross_entropy, gradient_descent_levenberg_marquardt> nn;

    nn << fully_connected_layer<sigmoid>(3, 2);

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
    nn.optimizer().alpha = 0.1;
    nn.train(data, train, 1, 10);

    vec_t predicted = nn.predict(a);

    EXPECT_NEAR(predicted[0], t[0], 1E-5);
    EXPECT_NEAR(predicted[1], t[1], 1E-5);

    predicted = nn.predict(a2);

    EXPECT_NEAR(predicted[0], t2[0], 1E-5);
    EXPECT_NEAR(predicted[1], t2[1], 1E-5);
}

TEST(fully_connected, serialize) {
    fully_connected_layer<tan_h> layer1(10, 10);
    fully_connected_layer<tan_h> layer2(10, 10);

    vec_t v(10);

    uniform_rand(v.begin(), v.end(), -1.0, 1.0);
    layer1.init_weight();

    std::ostringstream os;
    layer1.save(os);

    std::istringstream is(os.str());
    layer2.load(is);

    const vec_t& out1 = layer1.forward_propagation(v, 0);
    const vec_t& out2 = layer2.forward_propagation(v, 0);

    for (size_t i = 0; i < out1.size(); i++)
        EXPECT_NEAR(out1[i], out2[i], 1e-4);
}

TEST(fully_connected, bprop2) {
    network<mse, gradient_descent> nn;

    nn << fully_connected_layer<tan_h>(4, 6)
       << fully_connected_layer<tan_h>(6, 3);

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
    nn.optimizer().alpha = 0.01;
    nn.train(data, train, 1, 10);

    vec_t predicted = nn.predict(a);

    EXPECT_NEAR(predicted[0], t[0], 1E-4);
    EXPECT_NEAR(predicted[1], t[1], 1E-4);

    predicted = nn.predict(a2);

    EXPECT_NEAR(predicted[0], t2[0], 1E-4);
    EXPECT_NEAR(predicted[1], t2[1], 1E-4);
}

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

TEST(max_pool, gradient_check) { // sigmoid - cross-entropy
    typedef cross_entropy loss_func;
    typedef sigmoid activation;
    typedef network<loss_func, gradient_descent_levenberg_marquardt> network;

    network nn;
    nn << fully_connected_layer<activation>(3, 8)
       << max_pooling_layer<activation>(4, 2, 1, 2); // 4x2 => 2x1

    vec_t a(3, 0.0);
    for (int i = 0; i < 3; i++) a[i] = i;
    label_t t = 0;

    nn.init_weight();
    for (int i = 0; i < 24; i++) nn[0]->weight()[i] = i;

    EXPECT_TRUE(nn.gradient_check(&a, &t, 1, 1e-5, GRAD_CHECK_ALL));
}

template <typename T>
void serialization_test(const T& src, T& dst)
{
    EXPECT_FALSE(src.has_same_weights(dst, 1E-5));

    boost::filesystem::path tmp_path = boost::filesystem::unique_path();

    if (boost::filesystem::exists(tmp_path))
        throw nn_error("file exists");

    std::string tmp_file_path = tmp_path.string();

    // write
    {
        std::ofstream ofs(tmp_file_path.c_str());
        ofs << src;
    }

    // read
    {
        std::ifstream ifs(tmp_file_path.c_str());
        ifs >> dst;
    }

    boost::filesystem::remove(tmp_path); // remove temporary file

    EXPECT_TRUE(src.has_same_weights(dst, 1E-5));
}

TEST(read_write, fully_connected)
{
    fully_connected_layer<tan_h> l1(100, 100);
    fully_connected_layer<tan_h> l2(100, 100);

    l1.init_weight();
    l2.init_weight();

    serialization_test(l1, l2);
}

TEST(read_write, convolutional)
{
    typedef mse loss_func;
    typedef network<loss_func, gradient_descent_levenberg_marquardt> network;

    convolutional_layer<tan_h> l1(5, 5, 3, 1, 1);
    convolutional_layer<tan_h> l2(5, 5, 3, 1, 1);

    l1.init_weight();
    l2.init_weight();

    serialization_test(l1, l2);
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

TEST(lrn, cross) {
    lrn_layer<identity> lrn(1, 1, 3, 4, /*alpha=*/1.5, /*beta=*/2.0, norm_region::across_channels);

    tiny_cnn::float_t in[4] = { -1.0, 3.0, 2.0, 5.0 };
    tiny_cnn::float_t expected[4] =
    {
        -1.0/36.0,    // -1.0 / (1+0.5*(1*1+3*3))^2
        3.0/64.0,     //  3.0 / (1+0.5*(1*1+3*3+2*2))^2
        2.0/400.0,    //  2.0 / (1+0.5*(3*3+2*2+5*5))^2
        5.0/15.5/15.5 // 5.0 / (1+0.5*(2*2+5*5))^2
    };

    auto out = lrn.forward_propagation(vec_t(in, in + 4), 0);

    EXPECT_FLOAT_EQ(expected[0], out[0]);
    EXPECT_FLOAT_EQ(expected[1], out[1]);
    EXPECT_FLOAT_EQ(expected[2], out[2]);
    EXPECT_FLOAT_EQ(expected[3], out[3]);
}

int main(void) {
    RUN_ALL_TESTS();
}