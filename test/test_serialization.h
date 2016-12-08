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


namespace tiny_dnn {

TEST(serialization, serialize_conv) {
    network<sequential> net;

    std::string json = R"(
    {
        "nodes": [
            {
                "type": "conv<sigmoid>",
                "in_size" : {
                    "width": 20,
                    "height" : 20,
                    "depth" : 10
                },
                "window_width" : 5,
                "window_height" : 5,
                "out_channels" : 5,
                "connection_table" : {
                    "rows": 0,
                    "cols" : 0,
                    "connection" : "all"
                },
                "pad_type" : 1,
                "has_bias" : true,
                "w_stride" : 2,
                "h_stride" : 2
            }
        ]
    }
    )";

    net.from_json(json);

    EXPECT_EQ(net[0]->layer_type(), "conv");
    EXPECT_EQ(net[0]->in_shape()[0], shape3d(20,20,10));
    EXPECT_EQ(net[0]->in_shape()[1], shape3d(5, 5, 10*5));
    EXPECT_EQ(net[0]->out_shape()[0], shape3d(10,10,5));
}


TEST(serialization, serialize_maxpool) {
    network<sequential> net;

    std::string json = R"(  
    {
        "nodes": [
            {
                "type": "maxpool<softmax>",
                "in_size": {
                    "width": 10,
                    "height": 10,
                    "depth": 3
                },
                "pool_size_x": 2,
                "pool_size_y": 2,
                "stride_x": 1,
                "stride_y": 1,
                "pad_type": 1
            }
        ]
    }
    )";

    net.from_json(json);

    EXPECT_EQ(net[0]->layer_type(), "max-pool");
    EXPECT_EQ(net[0]->in_shape()[0], shape3d(10, 10, 3));
    EXPECT_EQ(net[0]->out_shape()[0], shape3d(10, 10, 3));
}


TEST(serialization, serialize_avepool) {
    network<sequential> net;

    std::string json = R"(  
    {
        "nodes": [
            {
                "type": "avepool<relu>",
                "in_size": {
                    "width": 10,
                    "height": 10,
                    "depth": 3
                },
                "pool_size_x": 2,
                "pool_size_y": 2,
                "stride_x": 2,
                "stride_y": 2,
                "pad_type": 0
            }
        ]
    }
    )";

    net.from_json(json);

    EXPECT_EQ(net[0]->layer_type(), "ave-pool");
    EXPECT_EQ(net[0]->in_shape()[0], shape3d(10, 10, 3));
    EXPECT_EQ(net[0]->out_shape()[0], shape3d(5, 5, 3));
}

TEST(serialization, serialize_batchnorm) {
    network<sequential> net;

    std::string json = R"(  
    {
        "nodes": [
            {
                "type": "batchnorm",
                "in_spatial_size": 3,
                "in_channels": 2,
                "epsilon": 0.001,
                "momentum": 0.8,
                "phase": 0,
                "mean": [
                    0,
                    0
                ],
                "variance": [
                    0,
                    0
                ]
            }
        ]
    }
    )";

    net.from_json(json);

    EXPECT_EQ(net[0]->layer_type(), "batch-norm");
    EXPECT_EQ(net[0]->in_shape()[0], shape3d(3, 1, 2));
    EXPECT_EQ(net[0]->out_shape()[0], shape3d(3, 1, 2));
    EXPECT_FLOAT_EQ(net.at<batch_normalization_layer>(0).epsilon(), 0.001f);
    EXPECT_FLOAT_EQ(net.at<batch_normalization_layer>(0).momentum(), 0.8f);
}

TEST(serialization, serialize_concat) {
    network<sequential> net;

    std::string json = R"(  
    {
        "nodes": [
            {
                "type": "concat",
                "in_size": [
                    {
                        "width": 2,
                        "height": 1,
                        "depth": 1
                    },
                    {
                        "width": 2,
                        "height": 1,
                        "depth": 2
                    },
                    {
                        "width": 2,
                        "height": 1,
                        "depth": 3
                    }
                ]
            }
        ]
    }
    )";

    net.from_json(json);

    EXPECT_EQ(net[0]->layer_type(), "concat");
    EXPECT_EQ(net[0]->in_shape()[0], shape3d(2, 1, 1));
    EXPECT_EQ(net[0]->in_shape()[1], shape3d(2, 1, 2));
    EXPECT_EQ(net[0]->in_shape()[2], shape3d(2, 1, 3));
    EXPECT_EQ(net[0]->out_shape()[0], shape3d(2, 1, 6));
}

TEST(serialization, serialize_dropout) {
    network<sequential> net;

    std::string json = R"(  
    {
        "nodes": [
            {
                "type": "dropout",
                "in_size": 3,
                "dropout_rate": 0.5,
                "phase": 1
            }
        ]
    }
    )";

    net.from_json(json);

    EXPECT_EQ(net[0]->layer_type(), "dropout");
    EXPECT_EQ(net[0]->in_shape()[0], shape3d(3, 1, 1));
    EXPECT_EQ(net[0]->out_shape()[0], shape3d(3, 1, 1));
    EXPECT_FLOAT_EQ(net.at<dropout_layer>(0).dropout_rate(), 0.5f);
}

TEST(serialization, serialize_fully) {
    network<sequential> net;

    std::string json = R"(  
    {
        "nodes": [
            {
                "type": "fully_connected<elu>",
                "in_size": 100,
                "out_size": 20,
                "has_bias": false
            }
        ]
    }
    )";

    net.from_json(json);

    EXPECT_EQ(net[0]->layer_type(), "fully-connected");
    EXPECT_EQ(net[0]->in_shape()[0], shape3d(100, 1, 1));
    EXPECT_EQ(net[0]->out_shape()[0], shape3d(20, 1, 1));
}

TEST(serialization, serialize_lrn) {
    network<sequential> net;

    std::string json = R"(  
    {
        "nodes": [
            {
                "type": "lrn<elu>",
                "in_shape": {
                    "width": 5,
                    "height": 4,
                    "depth": 6
                },
                "size": 3,
                "alpha": 0.5,
                "beta": 2.5,
                "region": 1
            }
        ]
    }
    )";

    net.from_json(json);

    EXPECT_EQ(net[0]->layer_type(), "lrn");
    EXPECT_EQ(net[0]->in_shape()[0], shape3d(5, 4, 6));
    EXPECT_EQ(net[0]->out_shape()[0], shape3d(5, 4, 6));
}

TEST(serialization, serialize_power) {
    network<sequential> net;

    std::string json = R"(  
    {
        "nodes": [
            {
                "type": "power",
                "in_size": {
                    "width": 3,
                    "height": 2,
                    "depth": 1
                },
                "factor": 0.5,
                "scale": 0.2
            }
        ]
    }
    )";

    net.from_json(json);

    EXPECT_EQ(net[0]->layer_type(), "power");
    EXPECT_EQ(net[0]->in_shape()[0], shape3d(3, 2, 1));
    EXPECT_EQ(net[0]->out_shape()[0], shape3d(3, 2, 1));
    EXPECT_FLOAT_EQ(net.at<power_layer>(0).factor(), 0.5f);
    EXPECT_FLOAT_EQ(net.at<power_layer>(0).scale(), 0.2f);
}

TEST(serialization, serialize_slice) {
    network<sequential> net;

    std::string json = R"(  
    {
        "nodes": [
            {
                "type": "slice",
                "in_size": {
                    "width": 3,
                    "height": 2,
                    "depth": 2
                },
                "slice_type": 1,
                "num_outputs": 2
            }
        ]
    }
    )";

    net.from_json(json);

    EXPECT_EQ(net[0]->layer_type(), "slice");
    EXPECT_EQ(net[0]->in_shape()[0], shape3d(3, 2, 2));
    EXPECT_EQ(net[0]->out_shape()[0], shape3d(3, 2, 1));
    EXPECT_EQ(net[0]->out_shape()[1], shape3d(3, 2, 1));
}

TEST(serialization, sequential_to_json) {
    network<sequential> net1, net2;

    net1 << fully_connected_layer<tan_h>(10, 100)
         << dropout_layer(100, 0.3f, net_phase::test)
         << fully_connected_layer<softmax>(100, 9)
         << convolutional_layer<tan_h>(3, 3, 3, 1, 1);

    auto json = net1.to_json();

    net2.from_json(json);

    EXPECT_EQ(net1.in_data_size(), net2.in_data_size());
    EXPECT_EQ(net1.layer_size(), net2.layer_size());

    EXPECT_EQ(net1[0]->in_shape(), net2[0]->in_shape());
    EXPECT_EQ(net1[1]->in_shape(), net2[1]->in_shape());
    EXPECT_EQ(net1[2]->in_shape(), net2[2]->in_shape());
    EXPECT_EQ(net1[3]->in_shape(), net2[3]->in_shape());

    EXPECT_EQ(net1[0]->layer_type(), net2[0]->layer_type());
    EXPECT_EQ(net1[1]->layer_type(), net2[1]->layer_type());
    EXPECT_EQ(net1[2]->layer_type(), net2[2]->layer_type());
    EXPECT_EQ(net1[3]->layer_type(), net2[3]->layer_type());

    EXPECT_FLOAT_EQ(net1.at<dropout_layer>(1).dropout_rate(), net2.at<dropout_layer>(1).dropout_rate());
}

TEST(serialization, sequential_model) {
    network<sequential> net1, net2;

    net1 << fully_connected_layer<tan_h>(10, 16)
         << average_pooling_layer<relu>(4, 4, 1, 2)
         << power_layer(shape3d(2,2,1),  0.5f);

    net1.init_weight();

    auto path = unique_path();
    net1.save(path, content_type::model);

    net2.load(path, content_type::model);

    for (int i = 0; i < 3; i++) {
        ASSERT_EQ(net1[i]->in_shape(), net2[i]->in_shape());
        ASSERT_EQ(net1[i]->out_shape(), net2[i]->out_shape());
        ASSERT_EQ(net1[i]->layer_type(), net2[i]->layer_type());
    }
}


TEST(serialization, sequential_weights) {
    network<sequential> net1, net2;
    vec_t data = {1,2,3,4,5,0};

    net1 << batch_normalization_layer(3,2,0.01f,0.99f,net_phase::train)
         << linear_layer<elu>(3*2, 2.0f, 0.5f);

    net1.init_weight();
    net1.at<batch_normalization_layer>(0).update_immidiately(true);
    net1.predict(data);
    net1.set_netphase(net_phase::test);

    auto path = unique_path();
    net1.save(path, content_type::weights_and_model);

    net2.load(path, content_type::weights_and_model);

    auto res1 = net1.predict(data);
    auto res2 = net2.predict(data);

    EXPECT_TRUE(net1.has_same_weights(net2, 1e-3f));

    for (int i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(res1[i], res2[i]);
    }
}

TEST(serialization, graph_model_and_weights) {
    network<graph> net1, net2;
    vec_t in = {1, 2, 3};

    fully_connected_layer<tan_h> f1(3, 4);
    slice_layer s1(shape3d(2,1,2), slice_type::slice_channels, 2);
    fully_connected_layer<softmax> f2(2, 2);
    fully_connected_layer<elu> f3(2, 2);
    elementwise_add_layer c4(2, 2);

    f1 << s1;
    s1 << (f2, f3) << c4;

    construct_graph(net1, {&f1}, {&c4});

    net1.init_weight();
    auto res1 = net1.predict(in);

    auto path = unique_path();

    net1.save(path, content_type::weights_and_model);

    net2.load(path, content_type::weights_and_model);

    auto res2 = net2.predict(in);

    EXPECT_FLOAT_EQ(res1[0], res2[0]);
    EXPECT_FLOAT_EQ(res1[1], res2[1]);
}

} // namespace tiny-dnn
