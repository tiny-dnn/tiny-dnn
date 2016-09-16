/*
    Copyright (c) 2016, Taiga Nomi
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

#include <string>

#include "picotest/picotest.h"
#include "testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

namespace tiny_dnn {

TEST(max_pool, read_write) {
    max_pooling_layer<tan_h> l1(100, 100, 5, 2);
    max_pooling_layer<tan_h> l2(100, 100, 5, 2);

    l1.init_weight();
    l2.init_weight();

    serialization_test(l1, l2);
}

TEST(max_pool, forward) {
    max_pooling_layer<identity> l(4, 4, 1, 2);
    vec_t in = {
        0, 1, 2, 3,
        8, 7, 5, 6,
        4, 3, 1, 2,
        0,-1,-2,-3
    };

    vec_t expected = {
        8, 6,
        4, 2
    };

    vec_t res = l.forward({ { in } })[0][0];

    for (size_t i = 0; i < expected.size(); i++) {
        EXPECT_FLOAT_EQ(expected[i], res[i]);
    }
}

TEST(max_pool, setup_tiny) {
    max_pooling_layer<identity> l(4, 4, 1, 2, 2, core::backend_t::tiny_dnn);

    EXPECT_EQ(l.parallelize(),           true);           // if layer can be parallelized
    EXPECT_EQ(l.in_channels(),           cnn_size_t(1));  // num of input tensors
    EXPECT_EQ(l.out_channels(),          cnn_size_t(2));  // num of output tensors
    EXPECT_EQ(l.in_data_size(),          cnn_size_t(16)); // size of input tensors
    EXPECT_EQ(l.out_data_size(),         cnn_size_t(4));  // size of output tensors
    EXPECT_EQ(l.in_data_shape().size(),  cnn_size_t(1));  // num of inputs shapes
    EXPECT_EQ(l.out_data_shape().size(), cnn_size_t(1));  // num of output shapes
    EXPECT_EQ(l.weights().size(),        cnn_size_t(0));  // the wieghts vector size
    EXPECT_EQ(l.weights_grads().size(),  cnn_size_t(0));  // the wieghts vector size
    EXPECT_EQ(l.inputs().size(),         cnn_size_t(1));  // num of input edges
    EXPECT_EQ(l.outputs().size(),        cnn_size_t(2));  // num of outpus edges
    EXPECT_EQ(l.in_types().size(),       cnn_size_t(1));  // num of input data types
    EXPECT_EQ(l.out_types().size(),      cnn_size_t(2));  // num of output data types
    EXPECT_EQ(l.fan_in_size(),           cnn_size_t(4));  // num of incoming connections
    EXPECT_EQ(l.fan_out_size(),          cnn_size_t(1));  // num of outgoing connections
    EXPECT_STREQ(l.layer_type().c_str(), "max-pool");     // string with layer type
}

TEST(max_pool, forward_stride_tiny) {
    max_pooling_layer<identity> l(4, 4, 1, 2, 2, core::backend_t::tiny_dnn);
    vec_t in = {
        0, 1, 2, 3,
        8, 7, 5, 6,
        4, 3, 1, 2,
        0,-1,-2,-3
    };

    vec_t expected = {
        8, 6,
        4, 2
    };

    vec_t res = l.forward({ {in} })[0][0];

    for (size_t i = 0; i < expected.size(); i++) {
        EXPECT_FLOAT_EQ(expected[i], res[i]);
    }
}

#ifdef CNN_USE_NNPACK
TEST(max_pool, forward_stride_nnp) {
    max_pooling_layer<identity> l(4, 4, 1, 2, 2, core::backend_t::nnpack);
    vec_t in = {
        0, 1, 2, 3,
        8, 7, 5, 6,
        4, 3, 1, 2,
        0,-1,-2,-3
    };

    vec_t expected = {
        8, 6,
        4, 2
    };

    vec_t res = l.forward({ {in} })[0][0];

    for (size_t i = 0; i < expected.size(); i++) {
        EXPECT_FLOAT_EQ(expected[i], res[i]);
    }
}
#endif

TEST(max_pool, forward_stride) {
    max_pooling_layer<identity> l(4, 4, 1, 2, 1);
    vec_t in = {
        0, 1, 2, 3,
        8, 7, 5, 6,
        4, 3, 1, 2,
        0,-1,-2,-3
    };

    vec_t expected = {
        8, 7, 6,
        8, 7, 6,
        4, 3, 2
    };

    vec_t res = l.forward({ { in } })[0][0];

    for (size_t i = 0; i < expected.size(); i++) {
        EXPECT_FLOAT_EQ(expected[i], res[i]);
    }
}

TEST(max_pool, backward) {
    max_pooling_layer<identity> l(4, 4, 1, 2);
    vec_t in = {
        0, 1, 2, 3,
        8, 7, 5, 6,
        4, 3, 1, 2,
        0,-1,-2,-3
    };

    vec_t out_grad = {
        1, 2,
        3, 4
    };

    vec_t in_grad_expected = {
        0, 0, 0, 0,
        1, 0, 0, 2,
        3, 0, 0, 4,
        0, 0, 0, 0
    };

    l.forward({ {in} })[0];
    vec_t in_grad = l.backward(std::vector<tensor_t>{ {out_grad}})[0][0];

    for (size_t i = 0; i < in_grad.size(); i++) {
        EXPECT_FLOAT_EQ(in_grad_expected[i], in_grad[i]);
    }
}

}  // namespace tiny_dnn
