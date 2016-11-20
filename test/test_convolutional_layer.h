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

TEST(convolutional, setup_internal) {

    convolutional_layer<sigmoid> l(5, 5, 3, 1, 2);

    EXPECT_EQ(l.parallelize(),           true);           // if layer can be parallelized
    EXPECT_EQ(l.in_channels(),           serial_size_t(3));  // num of input tensors
    EXPECT_EQ(l.out_channels(),          serial_size_t(2));  // num of output tensors
    EXPECT_EQ(l.in_data_size(),          serial_size_t(25)); // size of input tensors
    EXPECT_EQ(l.out_data_size(),         serial_size_t(18)); // size of output tensors
    EXPECT_EQ(l.in_data_shape().size(),  serial_size_t(1));  // number of inputs shapes
    EXPECT_EQ(l.out_data_shape().size(), serial_size_t(1));  // num of output shapes
    EXPECT_EQ(l.weights().size(),        serial_size_t(2));  // the wieghts vector size
    EXPECT_EQ(l.weights_grads().size(),  serial_size_t(2));  // the wieghts vector size
    EXPECT_EQ(l.inputs().size(),         serial_size_t(3));  // num of input edges
    EXPECT_EQ(l.outputs().size(),        serial_size_t(2));  // num of outpus edges
    EXPECT_EQ(l.in_types().size(),       serial_size_t(3));  // num of input data types
    EXPECT_EQ(l.out_types().size(),      serial_size_t(2));  // num of output data types
    EXPECT_EQ(l.fan_in_size(),           serial_size_t(9));  // num of incoming connections
    EXPECT_EQ(l.fan_out_size(),          serial_size_t(18)); // num of outgoing connections
    EXPECT_STREQ(l.layer_type().c_str(), "conv");         // string with layer type
}

inline void randomize_tensor(tensor_t& tensor)
{
    for (auto& vec : tensor) {
        uniform_rand(vec.begin(), vec.end(), -1.0f, 1.0f);
    }
}

// prepare tensor buffers for unit test
class tensor_buf {
public:
    tensor_buf(tensor_buf& other)
        : in_data_(other.in_data_), out_data_(other.out_data_) {
        for (auto& d : in_data_) in_ptr_.push_back(&d);
        for (auto& d : out_data_) out_ptr_.push_back(&d);
    }

    explicit tensor_buf(const layer& l, bool randomize = true)
    : in_data_(l.in_channels()), out_data_(l.out_channels()),
      in_ptr_(l.in_channels()),  out_ptr_(l.out_channels()) {

        for (size_t i = 0; i < l.in_channels(); i++) {
            in_data_[i].resize(1, vec_t(l.in_shape()[i].size()));
            in_ptr_[i] = &in_data_[i];
        }

        for (size_t i = 0; i < l.out_channels(); i++) {
            out_data_[i].resize(1, vec_t(l.out_shape()[i].size()));
            out_ptr_[i] = &out_data_[i];
        }

        if (randomize) {
            for (auto& tensor : in_data_)  randomize_tensor(tensor);
            for (auto& tensor : out_data_) randomize_tensor(tensor);
        }
    }

    tensor_t& in_at(size_t i) { return in_data_[i]; }
    tensor_t& out_at(size_t i) { return out_data_[i]; }

    std::vector<tensor_t*>& in_buf()  { return in_ptr_; }
    std::vector<tensor_t*>& out_buf() { return out_ptr_; }

private:
    std::vector<tensor_t> in_data_, out_data_;
    std::vector<tensor_t*> in_ptr_, out_ptr_;
};

TEST(convolutional, fprop) {

    convolutional_layer<sigmoid> l(5, 5, 3, 1, 2);

    tensor_buf buf(l, false);

    // short-hand references to the payload vectors
    vec_t &in     = buf.in_at(0)[0]
        , &out    = buf.out_at(0)[0]
        , &weight = buf.in_at(1)[0];

    ASSERT_EQ(l.in_shape()[1].size(), serial_size_t(18)); // weight

    uniform_rand(in.begin(), in.end(), -1.0, 1.0);

    l.setup(false);
    {
        l.forward_propagation(buf.in_buf(), buf.out_buf());

        for (auto o: out)
            EXPECT_DOUBLE_EQ(o, tiny_dnn::float_t(0.5));
    }

    weight[0] = 0.3f;  weight[1] = 0.1f; weight[2] = 0.2f;
    weight[3] = 0.0f;  weight[4] =-0.1f; weight[5] =-0.1f;
    weight[6] = 0.05f; weight[7] =-0.2f; weight[8] = 0.05f;

    weight[9]  = 0.0f; weight[10] =-0.1f; weight[11] = 0.1f;
    weight[12] = 0.1f; weight[13] =-0.2f; weight[14] = 0.3f;
    weight[15] = 0.2f; weight[16] =-0.3f; weight[17] = 0.2f;

    in[0] = 3;  in[1] = 2;  in[2] = 1;  in[3] = 5; in[4] = 2;
    in[5] = 3;  in[6] = 0;  in[7] = 2;  in[8] = 0; in[9] = 1;
    in[10] = 0; in[11] = 6; in[12] = 1; in[13] = 1; in[14] = 10;
    in[15] = 3; in[16] =-1; in[17] = 2; in[18] = 9; in[19] = 0;
    in[20] = 1; in[21] = 2; in[22] = 1; in[23] = 5; in[24] = 5;

    {
        l.forward_propagation(buf.in_buf(), buf.out_buf());

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

TEST(convolutional, with_stride) {
    /*
      forward - pass:

      [0,1,2,3,4,
       1,2,3,4,5,          [1,1,1,           [ 9.5,    18.5,
       2,3,4,5,6,  *  0.5*  1,1,1,  + 0.5 =
       3,4,5,6,7,           1,1,1]            18.5,    27.5]
       4,5,6,7,8]

    */

    float_t in[] = {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
        3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
        4.0f, 5.0f, 6.0f, 7.0f, 8.0f
    };

    float_t w[] = {
        0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f
    };

    float_t b[] = {
        0.5f
    };

    float_t expected_out[] = {
         9.5f, 18.5f,
        18.5f, 27.5f
    };

    convolutional_layer<identity> l(5, 5, 3, 1, 1, padding::valid, true, 2, 2);
    tensor_buf data(l, false), grad(l, false);

    data.in_at(0)[0] = vec_t(in, in + 25);
    data.in_at(1)[0] = vec_t(w, w + 9);
    data.in_at(2)[0] = vec_t(b, b + 1);

    l.forward_propagation(data.in_buf(), data.out_buf());

    vec_t& actual = data.out_at(0)[0];

    for (size_t i = 0; i < 4; i++) {
        ASSERT_DOUBLE_EQ(expected_out[i], actual[i]);
    }


    float_t curr_delta[] = {
        -1.0f, 2.0f,
        3.0f, 0.0f,
    };

    float_t expected_prev_delta[] = {
        -0.5f, -0.5f, 0.5f, 1.0f, 1.0f,
        -0.5f, -0.5f, 0.5f, 1.0f, 1.0f,
         1.0f,  1.0f, 2.0f, 1.0f, 1.0f,
         1.5f,  1.5f, 1.5f, 0.0f, 0.0f,
         1.5f,  1.5f, 1.5f, 0.0f, 0.0f
    };

    float_t expected_dw[] = {
        10.0f, 14.0f, 18.0f,
        14.0f, 18.0f, 22.0f,
        18.0f, 22.0f, 26.0f
    };

    float_t expected_db[] = {
        4.0f
    };

    grad.out_at(0)[0] = vec_t(curr_delta, curr_delta + 4);

    l.back_propagation(data.in_buf(), data.out_buf(), grad.out_buf(), grad.in_buf());

    const vec_t& actual_delta = grad.in_at(0)[0];
    const vec_t& actual_dw = grad.in_at(1)[0];
    const vec_t& actual_db = grad.in_at(2)[0];

    for (size_t i = 0; i < 25; i++) {
        ASSERT_DOUBLE_EQ(actual_delta[i], expected_prev_delta[i]);
    }
    for (size_t i = 0; i < 9; i++) {
        ASSERT_DOUBLE_EQ(actual_dw[i], expected_dw[i]);
    }
    ASSERT_DOUBLE_EQ(actual_db[0], expected_db[0]);

}

// test for AVX backends

#ifdef CNN_USE_AVX
TEST(convolutional, fprop_avx) {

    convolutional_layer<sigmoid> l(7, 7, 5, 1, 2);

    tensor_buf buf(l), buf2(l);

    l.set_backend_type(tiny_dnn::core::backend_t::internal);

    l.forward_propagation(buf.in_buf(), buf.out_buf());

    l.set_backend_type(tiny_dnn::core::backend_t::avx);

    l.forward_propagation(buf.in_buf(), buf2.out_buf());

    vec_t& out_avx = buf2.out_at(0)[0];
    vec_t& out_noavx = buf.out_at(0)[0];

    for (size_t i = 0; i < out_avx.size(); i++) {
        // check if all outputs between default backend and avx backend are the same
        EXPECT_NEAR(out_avx[i], out_noavx[i], 1E-5);
    }
}

TEST(convolutional, bprop_avx) {

    convolutional_layer<sigmoid> l(7, 7, 5, 1, 2);

    tensor_buf data(l), grad1(l);
    tensor_buf grad2(grad1);

    l.set_backend_type(tiny_dnn::core::backend_t::internal);

    l.forward_propagation(data.in_buf(), data.out_buf());
    l.back_propagation(data.in_buf(), data.out_buf(), grad1.out_buf(), grad1.in_buf());

    l.set_backend_type(tiny_dnn::core::backend_t::avx);

    l.forward_propagation(data.in_buf(), data.out_buf());
    l.back_propagation(data.in_buf(), data.out_buf(), grad2.out_buf(), grad2.in_buf());

    for (size_t ch = 0; ch < l.out_channels(); ch++) {
        vec_t& out_noavx = grad1.in_at(ch)[0];
        vec_t& out_avx = grad2.in_at(ch)[0];
        for (size_t i = 0; i < out_avx.size(); i++) {
            EXPECT_NEAR(out_avx[i], out_noavx[i], 1E-5);
        }
    }
}

TEST(convolutional, fprop_avx_1x1out) {

    convolutional_layer<sigmoid> l(5, 5, 5, 1, 2);

    tensor_buf buf(l), buf2(l);

    l.set_backend_type(tiny_dnn::core::backend_t::internal);

    l.forward_propagation(buf.in_buf(), buf.out_buf());

    l.set_backend_type(tiny_dnn::core::backend_t::avx);

    l.forward_propagation(buf.in_buf(), buf2.out_buf());

    vec_t& out_avx = buf2.out_at(0)[0];
    vec_t& out_noavx = buf.out_at(0)[0];

    for (size_t i = 0; i < out_avx.size(); i++) {
        EXPECT_NEAR(out_avx[i], out_noavx[i], 1E-5);
    }
}

TEST(convolutional, bprop_avx_1x1out) {

    convolutional_layer<sigmoid> l(5, 5, 5, 1, 2);

    tensor_buf data(l), grad1(l);
    tensor_buf grad2(grad1);

    l.set_backend_type(tiny_dnn::core::backend_t::internal);

    l.forward_propagation(data.in_buf(), data.out_buf());
    l.back_propagation(data.in_buf(), data.out_buf(), grad1.out_buf(), grad1.in_buf());

    l.set_backend_type(tiny_dnn::core::backend_t::avx);

    l.forward_propagation(data.in_buf(), data.out_buf());
    l.back_propagation(data.in_buf(), data.out_buf(), grad2.out_buf(), grad2.in_buf());

    for (size_t ch = 0; ch < l.out_channels(); ch++) {
        vec_t& out_noavx = grad1.in_at(ch)[0];
        vec_t& out_avx = grad2.in_at(ch)[0];
        for (size_t i = 0; i < out_avx.size(); i++) {
            EXPECT_NEAR(out_avx[i], out_noavx[i], 1E-5);
        }
    }
}

TEST(convolutional, fprop_avx_hstride) {

    convolutional_layer<sigmoid> l(7, 7, 5, 1, 2, padding::valid, true, 1, 2);

    tensor_buf buf(l), buf2(l);

    l.set_backend_type(tiny_dnn::core::backend_t::internal);

    l.forward_propagation(buf.in_buf(), buf.out_buf());

    l.set_backend_type(tiny_dnn::core::backend_t::avx);

    l.forward_propagation(buf.in_buf(), buf2.out_buf());

    vec_t& out_avx = buf2.out_at(0)[0];
    vec_t& out_noavx = buf.out_at(0)[0];

    for (size_t i = 0; i < out_avx.size(); i++) {
        EXPECT_NEAR(out_avx[i], out_noavx[i], 1E-5);
    }
}

TEST(convolutional, bprop_avx_hstride) {

    convolutional_layer<sigmoid> l(7, 7, 5, 1, 2, padding::valid, true, 1, 2);

    tensor_buf data(l), grad1(l);
    tensor_buf grad2(grad1);

    l.set_backend_type(tiny_dnn::core::backend_t::internal);

    l.forward_propagation(data.in_buf(), data.out_buf());
    l.back_propagation(data.in_buf(), data.out_buf(), grad1.out_buf(), grad1.in_buf());

    l.set_backend_type(tiny_dnn::core::backend_t::avx);

    l.forward_propagation(data.in_buf(), data.out_buf());
    l.back_propagation(data.in_buf(), data.out_buf(), grad2.out_buf(), grad2.in_buf());

    for (size_t ch = 0; ch < l.out_channels(); ch++) {
        vec_t& out_noavx = grad1.in_at(ch)[0];
        vec_t& out_avx = grad2.in_at(ch)[0];
        for (size_t i = 0; i < out_avx.size(); i++) {
            EXPECT_NEAR(out_avx[i], out_noavx[i], 1E-5);
        }
    }
}

TEST(convolutional, fprop_avx_hstride_1x1out) {

    convolutional_layer<sigmoid> l(5, 5, 5, 1, 2, padding::valid, true, 1, 2);

    tensor_buf buf(l), buf2(l);

    l.set_backend_type(tiny_dnn::core::backend_t::internal);

    l.forward_propagation(buf.in_buf(), buf.out_buf());

    l.set_backend_type(tiny_dnn::core::backend_t::avx);

    l.forward_propagation(buf.in_buf(), buf2.out_buf());

    vec_t& out_avx = buf2.out_at(0)[0];
    vec_t& out_noavx = buf.out_at(0)[0];

    for (size_t i = 0; i < out_avx.size(); i++) {
        EXPECT_NEAR(out_avx[i], out_noavx[i], 1E-5);
    }
}

TEST(convolutional, bprop_avx_hstride_1x1out) {

    convolutional_layer<sigmoid> l(5, 5, 5, 1, 2, padding::valid, true, 1, 2);

    tensor_buf data(l), grad1(l);
    tensor_buf grad2(grad1);

    l.set_backend_type(tiny_dnn::core::backend_t::internal);

    l.forward_propagation(data.in_buf(), data.out_buf());
    l.back_propagation(data.in_buf(), data.out_buf(), grad1.out_buf(), grad1.in_buf());

    l.set_backend_type(tiny_dnn::core::backend_t::avx);

    l.forward_propagation(data.in_buf(), data.out_buf());
    l.back_propagation(data.in_buf(), data.out_buf(), grad2.out_buf(), grad2.in_buf());

    for (size_t ch = 0; ch < l.out_channels(); ch++) {
        vec_t& out_noavx = grad1.in_at(ch)[0];
        vec_t& out_avx = grad2.in_at(ch)[0];
        for (size_t i = 0; i < out_avx.size(); i++) {
            EXPECT_NEAR(out_avx[i], out_noavx[i], 1E-5);
        }
    }
}

TEST(convolutional, fprop_avx_wstride) {

    convolutional_layer<sigmoid> l(7, 7, 5, 1, 2, padding::valid, true, 2, 1);

    tensor_buf buf(l), buf2(l);

    l.set_backend_type(tiny_dnn::core::backend_t::internal);

    l.forward_propagation(buf.in_buf(), buf.out_buf());

    l.set_backend_type(tiny_dnn::core::backend_t::avx);

    l.forward_propagation(buf.in_buf(), buf2.out_buf());

    vec_t& out_avx = buf2.out_at(0)[0];
    vec_t& out_noavx = buf.out_at(0)[0];

    for (size_t i = 0; i < out_avx.size(); i++) {
        EXPECT_NEAR(out_avx[i], out_noavx[i], 1E-5);
    }
}

TEST(convolutional, bprop_avx_wstride) {

    convolutional_layer<sigmoid> l(7, 7, 5, 1, 2, padding::valid, true, 2, 1);

    tensor_buf data(l), grad1(l);
    tensor_buf grad2(grad1);

    l.set_backend_type(tiny_dnn::core::backend_t::internal);

    l.forward_propagation(data.in_buf(), data.out_buf());
    l.back_propagation(data.in_buf(), data.out_buf(), grad1.out_buf(), grad1.in_buf());

    l.set_backend_type(tiny_dnn::core::backend_t::avx);

    l.forward_propagation(data.in_buf(), data.out_buf());
    l.back_propagation(data.in_buf(), data.out_buf(), grad2.out_buf(), grad2.in_buf());

    for (size_t ch = 0; ch < l.out_channels(); ch++) {
        vec_t& out_noavx = grad1.in_at(ch)[0];
        vec_t& out_avx = grad2.in_at(ch)[0];
        for (size_t i = 0; i < out_avx.size(); i++) {
            EXPECT_NEAR(out_avx[i], out_noavx[i], 1E-5);
        }
    }
}

#endif // CNN_USE_AVX

#ifdef CNN_USE_NNPACK
TEST(convolutional, fprop_nnp) {

    convolutional_layer<sigmoid> l(5, 5, 3, 1, 2,
        padding::valid, true, 1, 1, core::backend_t::nnpack);

    // layer::forward_propagation expects tensors, even if we feed only one input at a time
    auto create_simple_tensor = [](size_t vector_size) {
        return tensor_t(1, vec_t(vector_size));
    };

    // create simple tensors that wrap the payload vectors of the correct size
    tensor_t in_tensor     = create_simple_tensor(25)
           , out_tensor    = create_simple_tensor(18)
           , a_tensor      = create_simple_tensor(18)
           , weight_tensor = create_simple_tensor(18)
           , bias_tensor   = create_simple_tensor(2);

    // short-hand references to the payload vectors
    vec_t &in     = in_tensor[0]
        , &out    = out_tensor[0]
        , &weight = weight_tensor[0];

    ASSERT_EQ(l.in_shape()[1].size(), 18); // weight

    uniform_rand(in.begin(), in.end(), -1.0, 1.0);

    std::vector<tensor_t*> in_data, out_data;
    in_data.push_back(&in_tensor);
    in_data.push_back(&weight_tensor);
    in_data.push_back(&bias_tensor);
    out_data.push_back(&out_tensor);
    out_data.push_back(&a_tensor);
    l.setup(false);
    {
        l.forward_propagation(in_data, out_data);

        for (auto o: out)
            EXPECT_DOUBLE_EQ(o, tiny_dnn::float_t(0.5));
    }

    weight[0] = 0.3;  weight[1] = 0.1; weight[2] = 0.2;
    weight[3] = 0.0;  weight[4] =-0.1; weight[5] =-0.1;
    weight[6] = 0.05; weight[7] =-0.2; weight[8] = 0.05;

    weight[9]  = 0.0; weight[10] =-0.1; weight[11] = 0.1;
    weight[12] = 0.1; weight[13] =-0.2; weight[14] = 0.3;
    weight[15] = 0.2; weight[16] =-0.3; weight[17] = 0.2;

    in[0] = 3;  in[1] = 2;  in[2] = 1;  in[3] = 5; in[4] = 2;
    in[5] = 3;  in[6] = 0;  in[7] = 2;  in[8] = 0; in[9] = 1;
    in[10] = 0; in[11] = 6; in[12] = 1; in[13] = 1; in[14] = 10;
    in[15] = 3; in[16] =-1; in[17] = 2; in[18] = 9; in[19] = 0;
    in[20] = 1; in[21] = 2; in[22] = 1; in[23] = 5; in[24] = 5;

    {
        l.forward_propagation(in_data, out_data);

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
#endif

TEST(convolutional, gradient_check) { // tanh - mse
    network<sequential> nn;
    nn << convolutional_layer<tan_h>(5, 5, 3, 1, 1);

    const auto test_data = generate_gradient_check_data(nn.in_data_size());
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check<mse>(test_data.first,
                                       test_data.second,
                                       epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(convolutional, gradient_check2) { // sigmoid - mse
    network<sequential> nn;
    nn << convolutional_layer<sigmoid>(5, 5, 3, 1, 1);

    const auto test_data = generate_gradient_check_data(nn.in_data_size());
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check<mse>(test_data.first,
                                       test_data.second,
                                       epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(convolutional, gradient_check3) { // rectified - mse
    network<sequential> nn;

    nn << convolutional_layer<rectified_linear>(5, 5, 3, 1, 1);

    const auto test_data = generate_gradient_check_data(nn.in_data_size());
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check<mse>(test_data.first,
                                       test_data.second,
                                       epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(convolutional, gradient_check4) { // identity - mse
    network<sequential> nn;

    nn << convolutional_layer<identity>(5, 5, 3, 1, 1);

    const auto test_data = generate_gradient_check_data(nn.in_data_size());
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check<mse>(test_data.first,
                                       test_data.second,
                                       epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(convolutional, gradient_check5) { // sigmoid - cross-entropy
    network<sequential> nn;

    nn << convolutional_layer<sigmoid>(5, 5, 3, 1, 1);

    const auto test_data = generate_gradient_check_data(nn.in_data_size());
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check<cross_entropy>(test_data.first,
                                                 test_data.second,
                                                 epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(convolutional, gradient_check6) { // sigmoid - absolute
    network<sequential> nn;

    nn << convolutional_layer<sigmoid>(5, 5, 3, 1, 1);

    const auto test_data = generate_gradient_check_data(nn.in_data_size());
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check<absolute>(test_data.first,
                                            test_data.second,
                                            epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(convolutional, gradient_check7) { // sigmoid - absolute eps
    network<sequential> nn;

    nn << convolutional_layer<sigmoid>(5, 5, 3, 1, 1);

    const auto test_data = generate_gradient_check_data(nn.in_data_size());
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check<absolute_eps<100>>(test_data.first,
                                                     test_data.second,
                                                     epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(convolutional, gradient_check8_pad_same) { // sigmoid - mse - padding same
    network<sequential> nn;

    nn << convolutional_layer<sigmoid> (5, 5, 3, 1, 1, padding::same,
                                        true, 1, 1, core::backend_t::internal);

    const auto test_data = generate_gradient_check_data(nn.in_data_size());
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check<mse>(test_data.first,
                                       test_data.second,
                                       epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(convolutional, gradient_check9_w_stride) { // sigmoid - mse - w_stride > 1
    network<sequential> nn;

    nn << convolutional_layer<identity>(3, 3, 1, 1, 1, padding::valid,
                                        true, 2, 1, core::backend_t::internal);

    const auto test_data = generate_gradient_check_data(nn.in_data_size(), 1);
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check<mse>(test_data.first,
        test_data.second,
        epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(convolutional, gradient_check10_h_stride) { // sigmoid - mse - h_stride > 1
    network<sequential> nn;

    nn << convolutional_layer<identity>(3, 3, 1, 1, 1, padding::valid,
                                        true, 1, 2, core::backend_t::internal);

    const auto test_data = generate_gradient_check_data(nn.in_data_size(), 1);
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check<mse>(test_data.first,
        test_data.second,
        epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(convolutional, gradient_check11_connection_tbl) { // sigmoid - mse - has connection-tbl
    network<sequential> nn;
    bool tbl[3 * 3] = {
        true, false, true,
        false, true, false,
        true, true, false };

    connection_table connections(tbl, 3, 3);

    nn << convolutional_layer<sigmoid>(7, 7, 3, 3, 1, connections, padding::valid,
                                       true, 1, 1, core::backend_t::internal);

    const auto test_data = generate_gradient_check_data(nn.in_data_size());
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check<mse>(test_data.first,
        test_data.second,
        epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(convolutional, gradient_check12_pad_same) { // sigmoid - mse - padding same
	network<sequential> nn;

	nn << fully_connected_layer<identity>(10, 5*5) <<
		convolutional_layer<sigmoid>(5, 5, 3, 1, 1, padding::same,
		true, 1, 1, core::backend_t::internal);

	const auto test_data = generate_gradient_check_data(nn.in_data_size());
	nn.init_weight();
	EXPECT_TRUE(nn.gradient_check<mse>(test_data.first,
		test_data.second,
		epsilon<float_t>(), GRAD_CHECK_ALL));
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

TEST(convolutional, copy_and_pad_input_same) {

    conv_params params;
    params.in        = shape3d(5,5,1);
    params.weight    = shape3d(3,3,2);
    params.in_padded = shape3d(7,7,1);
    params.out       = shape3d(3,3,1);
    params.pad_type  = padding::same; // test target
    params.w_stride  = 1;
    params.h_stride  = 1;

    Conv2dPadding conv2d_padding(params);

    auto create_tensor = [](serial_size_t batch_size,
                            serial_size_t vector_size) {
        return tensor_t(batch_size, vec_t(vector_size));
    };

    tensor_t in_tensor = create_tensor(1, 1 * 5 * 5), out_tensor;

    fill_tensor(in_tensor, float_t(1));

    /* @in_tensor   --->   @out_tensor
     *
     *    1 1 1             0 0 0 0 0
     *    1 1 1             0 1 1 1 0
     *    1 1 1             0 1 1 1 0
     *                      0 1 1 1 0
     *                      0 0 0 0 0
     */

    conv2d_padding.copy_and_pad_input(in_tensor, out_tensor);

    EXPECT_EQ(out_tensor[0][7],  float_t(0));
    EXPECT_EQ(out_tensor[0][8],  float_t(1));
    EXPECT_EQ(out_tensor[0][9],  float_t(1));
    EXPECT_EQ(out_tensor[0][10], float_t(1));
    EXPECT_EQ(out_tensor[0][11], float_t(1));
    EXPECT_EQ(out_tensor[0][12], float_t(1));
    EXPECT_EQ(out_tensor[0][13], float_t(0));
}

TEST(convolutional, copy_and_unpad_delta_same) {

    conv_params params;
    params.in        = shape3d(3,3,1);
    params.weight    = shape3d(2,2,1);
    params.in_padded = shape3d(5,5,1);
    params.out       = shape3d(2,2,1);
    params.pad_type  = padding::same; // test target
    params.w_stride  = 1;
    params.h_stride  = 1;

    Conv2dPadding conv2d_padding(params);

    auto create_tensor = [](serial_size_t batch_size,
                            serial_size_t vector_size) {
        return tensor_t(batch_size, vec_t(vector_size));
    };

    tensor_t in_tensor = create_tensor(1, 1 * 5 * 5), out_tensor;

    fill_tensor(in_tensor, float_t(0));

    /*
     * @in_tensor   --->   @out_tensor
     *
     * 0 0 0 0 0             1 1 1
     * 0 1 1 1 0             1 1 1
     * 0 1 1 1 0             1 1 1
     * 0 1 1 1 0
     * 0 0 0 0 0
     *
     */

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            in_tensor[0][6 + 5*i + j] = float_t(1);
        }
    }

    conv2d_padding.copy_and_unpad_delta(in_tensor, out_tensor);

    for (serial_size_t i = 0; i < out_tensor[0].size(); ++i) {
        EXPECT_EQ(out_tensor[0][i], float_t(1));
    }
}

} // namespace tiny-dnn
