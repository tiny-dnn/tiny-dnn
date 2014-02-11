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
#include "tiny_cnn.h"
#include <boost/filesystem.hpp>

using namespace tiny_cnn;

TEST(convolutional, fprop) {
    typedef network<mse, gradient_descent_levenberg_marquardt> CNN;
    CNN nn;

    convolutional_layer<network<mse, gradient_descent_levenberg_marquardt>, sigmoid_activation> l(5, 5, 3, 1, 2);

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
    convolutional_layer<network<cross_entropy, gradient_descent_levenberg_marquardt>, sigmoid_activation> layer(5, 5, 3, 1, 1);

    nn.add(&layer);

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

    vec_t predicted;
    nn.predict(a, &predicted);
}

TEST(convolutional, gradient_check) { // tanh - mse
    network<mse, gradient_descent_levenberg_marquardt> nn;
    convolutional_layer<network<mse, gradient_descent_levenberg_marquardt>, tanh_activation> layer(5, 5, 3, 1, 1);
    nn.add(&layer);

    vec_t a(25, 0.0);
    label_t t = 3;

    uniform_rand(a.begin(), a.end(), -1, 1);
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check(&a, &t, 1, 1e-4, GRAD_CHECK_ALL));
}

TEST(convolutional, gradient_check2) { // sigmoid - mse
    network<mse, gradient_descent_levenberg_marquardt> nn;
    convolutional_layer<network<mse, gradient_descent_levenberg_marquardt>, sigmoid_activation> layer(5, 5, 3, 1, 1);
    nn.add(&layer);

    vec_t a(25, 0.0);
    label_t t = 3;

    uniform_rand(a.begin(), a.end(), -1, 1);
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check(&a, &t, 1, 1e-4, GRAD_CHECK_ALL));
}

TEST(convolutional, gradient_check3) { // rectified - mse
    network<mse, gradient_descent_levenberg_marquardt> nn;
    convolutional_layer<network<mse, gradient_descent_levenberg_marquardt>, rectified_linear> layer(5, 5, 3, 1, 1);
    nn.add(&layer);

    vec_t a(25, 0.0);
    label_t t = 3;

    uniform_rand(a.begin(), a.end(), -1, 1);
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check(&a, &t, 1, 1e-4, GRAD_CHECK_ALL));
}

TEST(convolutional, gradient_check4) { // identity - mse
    network<mse, gradient_descent_levenberg_marquardt> nn;
    convolutional_layer<network<mse, gradient_descent_levenberg_marquardt>, identity_activation> layer(5, 5, 3, 1, 1);
    nn.add(&layer);

    vec_t a(25, 0.0);
    label_t t = 3;

    uniform_rand(a.begin(), a.end(), -1, 1);
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check(&a, &t, 1, 1e-4, GRAD_CHECK_ALL));
}

TEST(convolutional, gradient_check5) { // sigmoid - cross-entropy
    network<cross_entropy, gradient_descent_levenberg_marquardt> nn;
    convolutional_layer<network<cross_entropy, gradient_descent_levenberg_marquardt>, sigmoid_activation> layer(5, 5, 3, 1, 1);
    nn.add(&layer);

    vec_t a(25, 0.0);
    label_t t = 3;

    uniform_rand(a.begin(), a.end(), -1, 1);
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check(&a, &t, 1, 1e-4, GRAD_CHECK_ALL));
}

TEST(convolutional, bprop2) {
    network<cross_entropy, gradient_descent_levenberg_marquardt> nn;
    convolutional_layer<network<cross_entropy, gradient_descent_levenberg_marquardt>, sigmoid_activation> layer(5, 5, 3, 1, 1);
    fully_connected_layer<network<cross_entropy, gradient_descent_levenberg_marquardt>, sigmoid_activation> layer2(9, 3);

    nn.add(&layer);
    nn.add(&layer2);

    vec_t a(25, 0.0), t(3, 0.0), a2(25, 0.0), t2(3, 0.0);

    for (int y = 0; y < 5; y++) {
        a[5*y+3] = 1.0;
    }


    t[0] = 0.0;
    t[1] = 0.5;
    t[2] = 1.0;

    uniform_rand(a2.begin(), a2.end(), -3, 3);
    uniform_rand(t2.begin(), t2.end(), 0, 1);

    std::vector<vec_t> data, train;

    for (int i = 0; i < 300; i++) {
        data.push_back(a);
        data.push_back(a2);
        train.push_back(t);
        train.push_back(t2);
    }
    nn.train(data, train);

    vec_t predicted;
    nn.predict(a, &predicted);
    nn.predict(a2, &predicted);
}

TEST(convolutional, serialize) {
    typedef network<mse, gradient_descent_levenberg_marquardt> NN;
    convolutional_layer<NN, tanh_activation> layer1(14, 14, 5, 1, 2);
    convolutional_layer<NN, tanh_activation> layer2(14, 14, 5, 1, 2);

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
    typedef network<mse, gradient_descent_levenberg_marquardt> NN;
#define O true
#define X false
    static const bool connection[] = {
        O, X, X, X, O, O, 
        O, O, X, X, X, O, 
        O, O, O, X, X, X
    };
#undef O
#undef X
    convolutional_layer<NN, tanh_activation> layer1(14, 14, 5, 3, 6, connection_table(connection, 3, 6));
    convolutional_layer<NN, tanh_activation> layer2(14, 14, 5, 3, 6, connection_table(connection, 3, 6));

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
    fully_connected_layer<network<cross_entropy, gradient_descent_levenberg_marquardt>, sigmoid_activation> layer(3, 2);
    nn.add(&layer);

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

    vec_t predicted;
    nn.predict(a, &predicted);

    EXPECT_NEAR(predicted[0], t[0], 1E-5);
    EXPECT_NEAR(predicted[1], t[1], 1E-5);

    nn.predict(a2, &predicted);

    EXPECT_NEAR(predicted[0], t2[0], 1E-5);
    EXPECT_NEAR(predicted[1], t2[1], 1E-5);
}

TEST(fully_connected, serialize) {
    typedef network<mse, gradient_descent_levenberg_marquardt> NN;
    fully_connected_layer<NN, tanh_activation> layer1(10, 10);
    fully_connected_layer<NN, tanh_activation> layer2(10, 10);

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
    fully_connected_layer<network<mse, gradient_descent>, tanh_activation> layer(4, 6);
    fully_connected_layer<network<mse, gradient_descent>, tanh_activation> layer2(6, 3);

    nn.add(&layer);
    nn.add(&layer2);

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

    vec_t predicted;
    nn.predict(a, &predicted);

    EXPECT_NEAR(predicted[0], t[0], 1E-4);
    EXPECT_NEAR(predicted[1], t[1], 1E-4);

    nn.predict(a2, &predicted);

    EXPECT_NEAR(predicted[0], t2[0], 1E-4);
    EXPECT_NEAR(predicted[1], t2[1], 1E-4);
}

TEST(multi_layer, gradient_check) { // sigmoid - cross-entropy
    typedef cross_entropy loss_func;
    typedef sigmoid_activation activation;
    typedef network<loss_func, gradient_descent_levenberg_marquardt> network;

    network nn;
    fully_connected_layer<network, activation> l1(10, 14*14*3);
    convolutional_layer<network, activation>   l2(14, 14, 5, 3, 6);
    average_pooling_layer<network, activation> l3(10, 10, 6, 2);
    fully_connected_layer<network, activation> l4(5*5*6, 3);

    nn.add(&l1);
    nn.add(&l2);
    nn.add(&l3);
    nn.add(&l4);

    vec_t a(10, 0.0);
    label_t t = 2;

    uniform_rand(a.begin(), a.end(), -1, 1);
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check(&a, &t, 1, 1e-4, GRAD_CHECK_RANDOM));
}

TEST(multi_layer, gradient_check2) { // tanh - mse
    typedef mse loss_func;
    typedef tanh_activation activation;
    typedef network<loss_func, gradient_descent_levenberg_marquardt> network;

    network nn;
    fully_connected_layer<network, activation> l1(10, 14*14*3);
    convolutional_layer<network, activation>   l2(14, 14, 5, 3, 6);
    average_pooling_layer<network, activation> l3(10, 10, 6, 2);
    fully_connected_layer<network, activation> l4(5*5*6, 3);

    nn.add(&l1);
    nn.add(&l2);
    nn.add(&l3);
    nn.add(&l4);

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
    fully_connected_layer<network, tanh_activation>     l1(10, 14*14*3);
    convolutional_layer<network, sigmoid_activation>    l2(14, 14, 5, 3, 6);
    average_pooling_layer<network, rectified_linear>    l3(10, 10, 6, 2);
    fully_connected_layer<network, identity_activation> l4(5*5*6, 3);

    nn.add(&l1);
    nn.add(&l2);
    nn.add(&l3);
    nn.add(&l4);

    vec_t a(10, 0.0);
    label_t t = 2;

    uniform_rand(a.begin(), a.end(), -1, 1);
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check(&a, &t, 1, 1e-4, GRAD_CHECK_RANDOM));
}

template <typename N>
void serialization_test(const layer_base<N>& src, layer_base<N>& dst)
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
	typedef mse loss_func;
	typedef network<loss_func, gradient_descent_levenberg_marquardt> network;

	fully_connected_layer<network, tanh_activation> l1(100, 100);
	fully_connected_layer<network, tanh_activation> l2(100, 100);

	l1.init_weight();
	l2.init_weight();

    serialization_test(l1, l2);
}

TEST(read_write, convolutional)
{
    typedef mse loss_func;
    typedef network<loss_func, gradient_descent_levenberg_marquardt> network;

    convolutional_layer<network, tanh_activation> l1(5, 5, 3, 1, 1);
    convolutional_layer<network, tanh_activation> l2(5, 5, 3, 1, 1);

    l1.init_weight();
    l2.init_weight();

    serialization_test(l1, l2);
}

int main(void) {
    RUN_ALL_TESTS();
}