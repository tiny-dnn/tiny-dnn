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

using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

class test_fc_layer : public fully_connected_layer<tan_h> {
public:
    typedef fully_connected_layer<tan_h> base;

    test_fc_layer() : base(10, 10) {
        ++counter();
    }

    test_fc_layer(const test_fc_layer& fc) : base(10, 10) {
        ++counter();
    }

    virtual ~test_fc_layer() {
        --counter();
    }

    test_fc_layer(test_fc_layer&& r) : base(std::move(r)){
        ++counter();
    }

    static int& counter() { static int i = 0; return i; }
};

TEST(network, construct_sequential_by_local_variables) {
    {
        network<sequential> net;
        test_fc_layer fc1, fc2;
        net << fc1 << fc2;
        ASSERT_EQ(test_fc_layer::counter(), 2);
    }
    ASSERT_EQ(test_fc_layer::counter(), 0);
}

TEST(network, construct_sequential_by_temporary_variables) {
    {
        network<sequential> net;
        net << test_fc_layer() << test_fc_layer();
        ASSERT_EQ(test_fc_layer::counter(), 2);
    }
    ASSERT_EQ(test_fc_layer::counter(), 0);
}

TEST(network, construct_sequential_by_shared_ptr) {
    {
        network<sequential> net;
        auto fc1 = std::make_shared<test_fc_layer>();
        auto fc2 = std::make_shared<test_fc_layer>();
        net << fc1 << fc2;
        ASSERT_EQ(test_fc_layer::counter(), 2);
    }
    ASSERT_EQ(test_fc_layer::counter(), 0);
}

TEST(network, construct_multi_by_local_variables) {
    network<sequential> net;
    conv<tan_h> conv1(32, 32, 5, 1, 6, padding::same);
    conv<sigmoid> conv2(32, 32, 7, 6, 12, padding::same);
    max_pool<relu> pool1(32, 32, 12, 2);
    lrn_layer<identity> lrn(16, 16, 4, 12);
    dropout dp(16*16*12, 0.5);
    fc<softmax> full(16*16*12, 1);

    net << conv1 << conv2 << pool1 << lrn << dp << full;
}

TEST(network, construct_multi_by_temporary_variables) {
    network<sequential> net;
    net << conv<tan_h>(32, 32, 5, 1, 6, padding::same)
        << conv<sigmoid>(32, 32, 7, 6, 12, padding::same)
        << max_pool<relu>(32, 32, 12, 2)
        << lrn_layer<identity>(16, 16, 4, 12)
        << dropout(16 * 16 * 12, 0.5)
        << fc<softmax>(16 * 16 * 12, 1);
}

TEST(network, in_dim) {
    network<sequential> net;
    convolutional_layer<identity> c1(32, 32, 5, 3, 6, padding::same);
    max_pooling_layer<identity> p1(32, 32, 6, 2);
    net << c1 << p1;

    EXPECT_EQ(c1.in_data_size(), net.in_data_size());
}

TEST(network, out_dim) {
    network<sequential> net;
    convolutional_layer<identity> c1(32, 32, 5, 3, 6, padding::same);
    max_pooling_layer<identity> p1(32, 32, 6, 2);
    net << c1 << p1;

    EXPECT_EQ(p1.out_data_size(), net.out_data_size());
}

TEST(network, name) {

    network<sequential> net1;
    network<sequential> net2("foo");

    EXPECT_EQ(net1.name(), "");
    EXPECT_EQ(net2.name(), "foo");
}

TEST(network, add) {
    network<sequential> net;
    net << convolutional_layer<identity>(32, 32, 5, 3, 6, padding::same);

    EXPECT_EQ(net.depth(), static_cast<serial_size_t>(1));
}

TEST(network, manual_init) {
    // initializing weights directly
    network<sequential> net;
    net << convolutional_layer<identity>(3, 3, 3, 1, 1)
        << fully_connected_layer<softmax>(1, 2, false);

    adagrad opt;

    vec_t* c1_w = net[0]->weights()[0];
    vec_t* c1_b = net[0]->weights()[1];
    vec_t* f1_w = net[1]->weights()[0];

    EXPECT_EQ(c1_w->size(), static_cast<serial_size_t>(9));
    EXPECT_EQ(c1_b->size(), static_cast<serial_size_t>(1));
    EXPECT_EQ(f1_w->size(), static_cast<serial_size_t>(2));

    *c1_w = { 0,1,2,3,4,5,6,7,8 };
    *c1_b = { 1 };
    *f1_w = { 1,2 };

    // check if the training and predicting works
    // https://github.com/tiny-dnn/tiny-dnn/issues/330
    net.predict({ 1,1,1,1,1,1,1,1,1 });

    net.train<mse, adagrad>(opt, tensor_t{ {1,1,1,1,1,1,1,1,1} }, tensor_t{ {1,2} }, 1, 1);
}

// TODO(nyanp): check out values again since the routine it's a bit sensitive

/*TEST(network, multi_out) {
    network<graph> net;
    adam optimizer;
    size_t tnum = 600;

    std::vector<vec_t> data;
    std::vector<std::vector<vec_t>> out;

    auto in = std::make_shared<input_layer>(shape3d(2,1,1));
    auto hidden = std::make_shared<layers::fc<tan_h>>(2,4);
    auto out1 = std::make_shared<layers::fc<tan_h>>(4,2);
    auto out2 = std::make_shared<layers::fc<tan_h>>(4,2);

    in << hidden << out1;
    hidden << out2;

    for (size_t i = 0; i < tnum; i++) {
        bool in[2] = { bernoulli(0.5), bernoulli(0.5) };
        // label_t expected = (in[0] ^ in[1]) ? 1 : 0;

        data.push_back({ static_cast<float_t>(in[0]),
                         static_cast<float_t>(in[1]) });

        out.emplace_back(std::vector<vec_t>{
            { static_cast<float_t>(in[0] && in[1]),
              static_cast<float_t>(in[0] || in[1]) }, // 1st output train and/or function
            { static_cast<float_t>(in[0] ^  in[1]),
              static_cast<float_t>(in[0] == in[1]) } // 2nd output train xor/eq function
        });
    }

    // construct single input, dual output network
    construct_graph(net, {in}, {out1,out2});

    optimizer.alpha *= 10;

    //net.fit<mse>(optimizer, data, out, 10, 10, nop, [&](){ std::cout << net.get_loss<mse>(data,out) << std::endl;});
    net.fit<mse>(optimizer, data, out, 10, 10);

    for (size_t i = 0; i < tnum; i++) {
        bool in[2] = { bernoulli(0.5), bernoulli(0.5) };
        std::vector<vec_t> actual = net.predict(
            std::vector<vec_t>{{ static_cast<float_t>(in[0]),
                                 static_cast<float_t>(in[1]) }});
        vec_t actual_out1 = actual[0];
        vec_t actual_out2 = actual[1];

        EXPECT_NEAR(actual_out1[0], in[0] && in[1], 0.1);
        EXPECT_NEAR(actual_out1[1], in[0] || in[1], 0.1);
        EXPECT_NEAR(actual_out2[0], in[0] ^  in[1], 0.1);
        EXPECT_NEAR(actual_out2[1], in[0] == in[1], 0.1);
    }
}*/

TEST(network, train_predict) {
    // train xor function
    network<sequential> net;
    adagrad optimizer;

    std::vector<vec_t> data;
    std::vector<label_t> label;
    size_t tnum = 300;

    optimizer.alpha *= 10;

    for (size_t i = 0; i < tnum; i++) {
        bool in[2] = { bernoulli(0.5), bernoulli(0.5) };
        data.push_back({ static_cast<float_t>(in[0]),
                         static_cast<float_t>(in[1]) });
        label.push_back((in[0] ^ in[1]) ? 1 : 0);
    }

    net << fully_connected_layer<tan_h>(2, 10)
        << fully_connected_layer<tan_h>(10, 2);

    net.train<mse>(optimizer, data, label, 10, 10);

    std::vector<tensor_t> parallel_input(tnum);
    std::vector<tensor_t> expected_parallel_output(tnum);

    for (size_t i = 0; i < tnum; i++) {
        const bool in[2] = { bernoulli(0.5), bernoulli(0.5) };
        const label_t expected = (in[0] ^ in[1]) ? 1 : 0;
        const vec_t input = { static_cast<float_t>(in[0]),
                              static_cast<float_t>(in[1]) };
        const label_t actual = net.predict_label(input);
        EXPECT_EQ(expected, actual);

        const auto actual_vec = net.predict(input);
        EXPECT_EQ(expected == 1, actual_vec[1] > actual_vec[0]);

        parallel_input[i] = tensor_t{ input };
        expected_parallel_output[i] = tensor_t{ actual_vec };
    }

    // test predicting multiple samples in parallel
    const auto actual_parallel_output = net.predict(parallel_input);
    for (size_t i = 0; i < tnum; i++) {
        EXPECT_NEAR(expected_parallel_output[i][0][0], actual_parallel_output[i][0][0], 1e-10);
        EXPECT_NEAR(expected_parallel_output[i][0][1], actual_parallel_output[i][0][1], 1e-10);
    }
}

TEST(network, set_netphase) {
    // TODO: add unit-test for public api
}

TEST(network, test) {
    network<sequential> net;
    fully_connected_layer<identity> fc(30, 1);
    int data_num = 300;

    net << fc;
    net.weight_init(weight_init::constant(1.0));
    net.init_weight();

    std::vector<vec_t> in, expected;

    for (int i = 0; i < data_num; i++) {
        vec_t v(30);
        uniform_rand(v.begin(), v.end(), -1.0, 1.0);
        float_t sum = std::accumulate(v.begin(), v.end(), (float_t)0.0);

        in.emplace_back(v);
        expected.emplace_back(vec_t{sum});
    }

    auto out = net.test(in);
    for (int i = 0; i < data_num; i++) { 
        for (size_t j = 0; j < out[i].size(); j++)
            EXPECT_FLOAT_EQ(out[i][j], expected[i][0]);
    }
}

TEST(network, get_loss) {
    // TODO: add unit-test for public api
}

TEST(network, at) {
    network<sequential> net;
    convolutional_layer<identity> c1(32, 32, 5, 3, 6, padding::same);
    average_pooling_layer<identity> p1(32, 32, 6, 2);

    net << c1 << p1;
    net.init_weight();

    // auto& c = net.at<convolutional_layer<identity>>(0);
    // auto& p = net.at<average_pooling_layer<identity>>(1);
}

TEST(network, bracket_operator) {
    network<sequential> net;

    net << convolutional_layer<identity>(32, 32, 5, 3, 6, padding::same)
        << average_pooling_layer<identity>(32, 32, 6, 2);

    EXPECT_EQ(net[0]->layer_type(), "conv");
    EXPECT_EQ(net[1]->layer_type(), "ave-pool");
}


TEST(network, weight_init) {
    network<sequential> net;

    net << convolutional_layer<identity>(32, 32, 5, 3, 6, padding::same)
        << average_pooling_layer<identity>(32, 32, 6, 2);

    // change all layers at once
    net.weight_init(weight_init::constant(2.0));
    net.init_weight();

    vec_t& w1 = *net[0]->weights()[0];
    vec_t& w2 = *net[1]->weights()[0];

    for (size_t i = 0; i < w1.size(); i++)
        EXPECT_NEAR(w1[i], 2.0, 1e-10);

    for (size_t i = 0; i < w2.size(); i++)
        EXPECT_NEAR(w2[i], 2.0, 1e-10);
}

TEST(network, weight_init_per_layer) {
    network<sequential> net;

    net << convolutional_layer<identity>(32, 32, 5, 3, 6, padding::same)
        << average_pooling_layer<identity>(32, 32, 6, 2);

    // change specific layer
    net[0]->weight_init(weight_init::constant(2.0));
    net[1]->weight_init(weight_init::constant(1.0));
    net.init_weight();

    vec_t& w1 = *net[0]->weights()[0];
    vec_t& w2 = *net[1]->weights()[0];

    for (size_t i = 0; i < w1.size(); i++)
        EXPECT_NEAR(w1[i], 2.0, 1e-10);

    for (size_t i = 0; i < w2.size(); i++)
        EXPECT_NEAR(w2[i], 1.0, 1e-10);
}

TEST(network, bias_init) {
    network<sequential> net;

    net << convolutional_layer<identity>(32, 32, 5, 3, 6, padding::same)
        << average_pooling_layer<identity>(32, 32, 6, 2);

    net.bias_init(weight_init::constant(2.0));
    net.init_weight();

    vec_t& w1 = *net[0]->weights()[1];
    vec_t& w2 = *net[1]->weights()[1];

    for (size_t i = 0; i < w1.size(); i++)
        EXPECT_NEAR(w1[i], 2.0, 1e-10);

    for (size_t i = 0; i < w2.size(); i++)
        EXPECT_NEAR(w2[i], 2.0, 1e-10);
}

TEST(network, bias_init_per_layer) {
    network<sequential> net;

    net << convolutional_layer<identity>(32, 32, 5, 3, 6, padding::same)
        << average_pooling_layer<identity>(32, 32, 6, 2);

    net[0]->bias_init(weight_init::constant(2.0));
    net[1]->bias_init(weight_init::constant(1.0));
    net.init_weight();

    vec_t& w1 = *net[0]->weights()[1];
    vec_t& w2 = *net[1]->weights()[1];

    for (size_t i = 0; i < w1.size(); i++)
        EXPECT_NEAR(w1[i], 2.0, 1e-10);

    for (size_t i = 0; i < w2.size(); i++)
        EXPECT_NEAR(w2[i], 1.0, 1e-10);
}

TEST(network, gradient_check) { // sigmoid - cross-entropy
    typedef cross_entropy loss_func;
    typedef sigmoid activation;
    typedef network<sequential> network;

    network nn;
    nn << fully_connected_layer<activation>(10, 14*14*3)
       << convolutional_layer<activation>(14, 14, 5, 3, 6)
       << average_pooling_layer<activation>(10, 10, 6, 2)
       << fully_connected_layer<activation>(5*5*6, 3);

    const auto test_data = generate_gradient_check_data(nn.in_data_size());
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check<loss_func>(test_data.first,
                                             test_data.second,
                                             epsilon<float_t>(), GRAD_CHECK_RANDOM));
}

TEST(network, gradient_check2) { // tan_h - mse
    typedef mse loss_func;
    typedef tan_h activation;
    typedef network<sequential> network;

    network nn;
    nn << fully_connected_layer<activation>(10, 14 * 14 * 3)
       << convolutional_layer<activation>(14, 14, 5, 3, 6)
       << average_pooling_layer<activation>(10, 10, 6, 2)
       << fully_connected_layer<activation>(5 * 5 * 6, 3);

    const auto test_data = generate_gradient_check_data(nn.in_data_size());
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check<loss_func>(test_data.first,
                                             test_data.second,
                                             epsilon<float_t>(), GRAD_CHECK_RANDOM));
}

TEST(network, gradient_check3) { // mixture - mse
    typedef mse loss_func;
    typedef network<sequential> network;

    network nn;
    nn << fully_connected_layer<tan_h>(10, 14 * 14 * 3)
       << convolutional_layer<sigmoid>(14, 14, 5, 3, 6)
       << average_pooling_layer<rectified_linear>(10, 10, 6, 2)
       << fully_connected_layer<identity>(5 * 5 * 6, 3);

    const auto test_data = generate_gradient_check_data(nn.in_data_size());
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check<loss_func>(test_data.first,
                                             test_data.second,
                                             epsilon<float_t>(), GRAD_CHECK_RANDOM));
}

TEST(network, gradient_check4) { // sigmoid - cross-entropy
    typedef cross_entropy loss_func;
    typedef sigmoid activation;
    typedef network<sequential> network;

    network nn;
    nn << fully_connected_layer<activation>(10, 14 * 14 * 3)
       << convolutional_layer<activation>(14, 14, 5, 3, 6)
       << average_pooling_layer<activation>(10, 10, 6, 2)
       << fully_connected_layer<activation>(5 * 5 * 6, 3);

    const auto test_data = generate_gradient_check_data(nn.in_data_size());
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check<loss_func>(test_data.first,
                                             test_data.second,
                                             epsilon<float_t>(), GRAD_CHECK_RANDOM));
}

TEST(network, gradient_check5) { // softmax - cross-entropy
    typedef cross_entropy loss_func;
    typedef softmax activation;
    typedef network<sequential> network;

    network nn;
    nn << fully_connected_layer<activation>(10, 14 * 14 * 3)
       << convolutional_layer<activation>(14, 14, 5, 3, 6)
       << average_pooling_layer<activation>(10, 10, 6, 2)
       << fully_connected_layer<activation>(5 * 5 * 6, 3);

    const auto test_data = generate_gradient_check_data(nn.in_data_size());
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check<loss_func>(test_data.first,
                                             test_data.second,
                                             1e-1f, GRAD_CHECK_RANDOM));
}

TEST(network, gradient_check6) { // sigmoid - cross-entropy
    typedef cross_entropy loss_func;
    typedef sigmoid activation;
    typedef network<sequential> network;

    network nn;
    nn << fully_connected_layer<activation>(3, 201)
       << fully_connected_layer<activation>(201, 2);

    const auto test_data = generate_gradient_check_data(nn.in_data_size());
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check<loss_func>(test_data.first,
                                             test_data.second,
                                             epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(network, gradient_check7) { // leaky-relu - mse
    typedef mse loss_func;
    typedef leaky_relu activation;

    auto nn = make_mlp<activation>({ 3, 201, 2 });

    const auto test_data = generate_gradient_check_data(nn.in_data_size());
    nn.init_weight();
	// We need to use larger threshold here, because
	// relu/elu/leaky-relu has non-smooth region of the form
	// 
	// @todo improve gradient-checker to ignore non-smooth region like
	//       caffe's GradientChecker (they have kink/kink-range parameter to handle it)
    EXPECT_TRUE(nn.gradient_check<loss_func>(test_data.first,
                                             test_data.second,
                                             10*epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(network, gradient_check8) { // elu - mse
    typedef mse loss_func;
    typedef elu activation;

    auto nn = make_mlp<activation>({3, 201, 2});

    const auto test_data = generate_gradient_check_data(nn.in_data_size());
    nn.init_weight();

	// We need to use larger threshold here, because
	// relu/elu/leaky-relu has non-smooth region of the form
	// 
	// @todo improve gradient-checker to ignore non-smooth region like
	//       caffe's GradientChecker (they have kink/kink-range parameter to handle it)
    EXPECT_TRUE(nn.gradient_check<loss_func>(test_data.first,
                                             test_data.second,
                                             10*epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(network, gradient_check9) { // tan_hp1m2 - mse
    typedef mse loss_func;
    typedef tan_hp1m2 activation;

    auto nn = make_mlp<activation>({ 3, 201, 2 });

    const auto test_data = generate_gradient_check_data(nn.in_data_size());
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check<loss_func>(test_data.first,
        test_data.second,
        epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(network, read_write)
{
    typedef mse loss_func;
    typedef network<sequential> network;

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
    adagrad optimizer;
    n1.train<loss_func>(optimizer, t, l, 1, 1);

    serialization_test(n1, n2);

    vec_t in(32*32, 0.0);

    auto res1 = n1.predict(in);
    auto res2 = n2.predict(in);

    ASSERT_TRUE(n1.has_same_weights(n2, epsilon<float_t>()));

    for (int i = 0; i < 10; i++) {
        tiny_dnn::float_t eps = std::abs(res1[i]) * 1e-5f;
        ASSERT_TRUE(std::abs(res1[i] - res2[i]) < eps);
    }
}

TEST(network, trainable) {
    auto net = make_mlp<sigmoid>({ 2,3,2,1 }); // fc(2,3) - fc(3,2) - fc(2,1)

    // trainable=false, or "freeze" 2nd layer fc(3,2)
    net[1]->set_trainable(false);

    vec_t w0 = { 0,1,2,3,4,5 };
    vec_t w1 = { 6,7,8,9,8,7 };
    vec_t w2 = { 6,5 };

    *net[0]->weights()[0] = { 0,1,2,3,4,5 };
    *net[1]->weights()[0] = { 6,7,8,9,8,7 };
    *net[2]->weights()[0] = { 6,5 };

    adam a;

    net.init_weight();

    auto w0_standby = *net[0]->weights()[0];
    auto w1_standby = *net[1]->weights()[0];
    auto w2_standby = *net[2]->weights()[0];

    EXPECT_NE(w0, w0_standby);
    EXPECT_EQ(w1, w1_standby);
    EXPECT_NE(w2, w2_standby);

    std::vector<vec_t> data{ {1,0}, {0,2} };
    std::vector<vec_t> out{ {2}, {1} };

    net.fit<mse>(a, data, out, 1, 1);

    auto w0_after_update = *net[0]->weights()[0];
    auto w1_after_update = *net[1]->weights()[0];
    auto w2_after_update = *net[2]->weights()[0];

    EXPECT_NE(w0, w0_after_update);
    EXPECT_EQ(w1, w1_after_update);
    EXPECT_NE(w2, w2_after_update);
}

} // namespace tiny-dnn
