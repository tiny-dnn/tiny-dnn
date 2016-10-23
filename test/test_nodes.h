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
 #include "gtest/gtest.h"
#include "testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn::layers;

namespace tiny_dnn {

TEST(nodes, sequential) {
    network<sequential> nn;

    nn << fc<tan_h>(10, 100)
       << fc<softmax>(100, 10);
}

TEST(nodes, graph_no_branch) {
    // declare nodes
    auto in = std::make_shared<input_layer>(shape3d(8, 8, 1));

    auto cnn = std::make_shared<
        convolutional_layer<tan_h> >(8, 8, 3, 1, 4);

    auto pool = std::make_shared<
        average_pooling_layer<tan_h> >(6, 6, 4, 2);

    auto out = std::make_shared<linear_layer<relu> >(3 * 3 * 4);

    // connect
    in << cnn << pool << out;

    network<graph> net;
    construct_graph(net, { in }, { out });
}

TEST(nodes, graph_branch) {
    // declare nodes
    auto in1 = std::make_shared<input_layer>(shape3d(3, 1, 1));
    auto in2 = std::make_shared<input_layer>(shape3d(3, 1, 1));
    auto added = std::make_shared<add>(2, 3);
    auto out = std::make_shared<linear_layer<relu>>(3);

    // connect
    (in1, in2) << added;
    added << out;

    network<graph> net;
    construct_graph(net, { in1, in2 }, { out });

    auto res = net.predict({ { 2,4,3 },{ -1,2,-5 } })[0];

    // relu({2,4,3} + {-1,2,-5}) = {1,6,0}
    EXPECT_FLOAT_EQ(static_cast<float_t>(res[0]), static_cast<float_t>(1.0));
    EXPECT_FLOAT_EQ(static_cast<float_t>(res[1]), static_cast<float_t>(6.0));
    EXPECT_FLOAT_EQ(static_cast<float_t>(res[2]), static_cast<float_t>(0.0));
}

TEST(nodes, graph_branch2) {
    // declare nodes
    input_layer in1(shape3d(3, 1, 1));
    input_layer in2(shape3d(3, 1, 1));
    add added(2, 3);
    linear_layer<relu> out(3);

    // connect
    (in1, in2) << added;
    added << out;

    network<graph> net;
    construct_graph(net, { &in1, &in2 }, { &out });

    auto res = net.predict({ { 2,4,3 },{ -1,2,-5 } })[0];

    // relu({2,4,3} + {-1,2,-5}) = {1,6,0}
    EXPECT_FLOAT_EQ(static_cast<float_t>(res[0]), static_cast<float_t>(1.0));
    EXPECT_FLOAT_EQ(static_cast<float_t>(res[1]), static_cast<float_t>(6.0));
    EXPECT_FLOAT_EQ(static_cast<float_t>(res[2]), static_cast<float_t>(0.0));
}

} // namespace tiny-dnn
