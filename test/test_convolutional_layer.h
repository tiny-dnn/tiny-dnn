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
#include "tiny_dnn/tiny_dnn.h"

namespace tiny_dnn {

TEST(convolutional, setup_tiny) {

    convolutional_layer<sigmoid> l(5, 5, 3, 1, 2);

    EXPECT_EQ(l.parallelize(),           true);           // if layer can be parallelized
    EXPECT_EQ(l.in_channels(),           cnn_size_t(3));  // num of input tensors
    EXPECT_EQ(l.out_channels(),          cnn_size_t(2));  // num of output tensors
    EXPECT_EQ(l.in_data_size(),          cnn_size_t(25)); // size of input tensors
    EXPECT_EQ(l.out_data_size(),         cnn_size_t(18)); // size of output tensors
    EXPECT_EQ(l.in_data_shape().size(),  cnn_size_t(1));  // number of inputs shapes
    EXPECT_EQ(l.out_data_shape().size(), cnn_size_t(1));  // num of output shapes
    EXPECT_EQ(l.weights().size(),        cnn_size_t(2));  // the wieghts vector size
    EXPECT_EQ(l.weights_grads().size(),  cnn_size_t(2));  // the wieghts vector size
    EXPECT_EQ(l.inputs().size(),         cnn_size_t(3));  // num of input edges
    EXPECT_EQ(l.outputs().size(),        cnn_size_t(2));  // num of outpus edges
    EXPECT_EQ(l.in_types().size(),       cnn_size_t(3));  // num of input data types
    EXPECT_EQ(l.out_types().size(),      cnn_size_t(2));  // num of output data types
    EXPECT_EQ(l.fan_in_size(),           cnn_size_t(9));  // num of incoming connections
    EXPECT_EQ(l.fan_out_size(),          cnn_size_t(18)); // num of outgoing connections
    EXPECT_STREQ(l.layer_type().c_str(), "conv");         // string with layer type
    EXPECT_TRUE(l.engine() == backend_t::tiny_dnn);       // the engine type
}

TEST(convolutional, fprop) {

    convolutional_layer<sigmoid> l(5, 5, 3, 1, 2);

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

    ASSERT_EQ(l.in_shape()[1].size(), cnn_size_t(18)); // weight

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
                                        true, 1, 1, core::backend_t::tiny_dnn);

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


TEST(convolutional, copy_and_pad_input_valid) {

    conv_params params;
    params.in        = shape3d(5,5,1);
    params.weight    = shape3d(3,3,2);
    params.in_padded = shape3d(5,5,1);
    params.out       = shape3d(3,3,1);
    params.pad_type  = padding::valid; // test target
    params.w_stride  = 1;
    params.h_stride  = 1;

    Conv2d conv2d_op;
           conv2d_op.setParams(&params);

    auto create_tensor = [](cnn_size_t batch_size,
                            cnn_size_t vector_size) {
        return tensor_t(batch_size, vec_t(vector_size));
    };

    tensor_t in_tensor = create_tensor(1, 1 * 5 * 5), out_tensor;

    fill_tensor(in_tensor, float_t(1));

    conv2d_op.copy_and_pad_input(in_tensor, out_tensor);

    for (cnn_size_t i = 0; i < out_tensor[0].size(); ++i) {
        EXPECT_EQ(out_tensor[0][i], float_t(1));
    }
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

    Conv2d conv2d_op;
           conv2d_op.setParams(&params);

    auto create_tensor = [](cnn_size_t batch_size,
                            cnn_size_t vector_size) {
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

    conv2d_op.copy_and_pad_input(in_tensor, out_tensor);

    EXPECT_EQ(out_tensor[0][7],  float_t(0));
    EXPECT_EQ(out_tensor[0][8],  float_t(1));
    EXPECT_EQ(out_tensor[0][9],  float_t(1));
    EXPECT_EQ(out_tensor[0][10], float_t(1));
    EXPECT_EQ(out_tensor[0][11], float_t(1));
    EXPECT_EQ(out_tensor[0][12], float_t(1));
    EXPECT_EQ(out_tensor[0][13], float_t(0));
}

TEST(convolutional, copy_and_unpad_valid) {

    conv_params params;
    params.in        = shape3d(5,5,1);
    params.weight    = shape3d(3,3,2);
    params.in_padded = shape3d(5,5,1);
    params.out       = shape3d(3,3,1);
    params.pad_type  = padding::valid; // test target
    params.w_stride  = 1;
    params.h_stride  = 1;

    Conv2d conv2d_op;
           conv2d_op.setParams(&params);

    auto create_tensor = [](cnn_size_t batch_size,
                            cnn_size_t vector_size) {
        return tensor_t(batch_size, vec_t(vector_size));
    };

    tensor_t in_tensor = create_tensor(1, 1 * 7 * 7), out_tensor;

    fill_tensor(in_tensor, float_t(1));

    conv2d_op.copy_and_unpad_delta(in_tensor, out_tensor);

    for (cnn_size_t i = 0; i < out_tensor[0].size(); ++i) {
        EXPECT_EQ(out_tensor[0][i], float_t(1));
    }
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

    Conv2d conv2d_op;
           conv2d_op.setParams(&params);

    auto create_tensor = [](cnn_size_t batch_size,
                            cnn_size_t vector_size) {
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

    conv2d_op.copy_and_unpad_delta(in_tensor, out_tensor);

    for (cnn_size_t i = 0; i < out_tensor[0].size(); ++i) {
        EXPECT_EQ(out_tensor[0][i], float_t(1));
    }
}

} // namespace tiny-dnn
