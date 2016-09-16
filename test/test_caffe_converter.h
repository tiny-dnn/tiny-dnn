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

inline std::shared_ptr<network<sequential>>
create_net_from_json(const std::string& caffemodeljson, const shape3d& shape = shape3d()) {
    std::string tmp_file_path = unique_path();

    {
        std::ofstream ofs(tmp_file_path.c_str());
        ofs << caffemodeljson;
    }
    auto model = create_net_from_caffe_prototxt(tmp_file_path, shape);

    std::remove(tmp_file_path.c_str());

    return model;
}

TEST(caffe_converter, rectangle_input) {
    std::string json = R"(
    name: "RectangleNet"
    input: "data"
    input_shape {
      dim: 1
      dim: 1
      dim: 40
      dim: 24
    }
    layer {
      name: "conv1"
      type: "Convolution"
      bottom: "data"
      top: "conv1"
      convolution_param {
        num_output: 96
        kernel_size: 3
      }
    }
    layer {
      name: "fc1"
      type: "InnerProduct"
      bottom: "conv1"
      top: "fc1"
      inner_product_param {
        num_output: 10
      }
    }
    )";


    auto model = create_net_from_json(json);

    // conv->pool->conv->pool->fc->relu->fc->softmax
    ASSERT_EQ(model->depth(), 2);

    EXPECT_EQ((*model)[0]->in_shape()[0], shape3d(24, 40, 1));
    EXPECT_EQ((*model)[0]->out_shape()[0], shape3d(22, 38, 96));


    EXPECT_EQ((*model)[1]->in_shape()[0], shape3d(22*38*96, 1, 1));
    EXPECT_EQ((*model)[1]->out_shape()[0], shape3d(10, 1, 1));

}

/**
 * test if we can parse lenet-model, defined in caffe/examples/mnsit
 **/
TEST(caffe_converter, lenet) {
    std::string json = R"(
    name: "LeNet"
    layer {
      name: "mnist"
      type: "Data"
      top: "data"
      top: "label"
      include {
        phase: TRAIN
      }
      transform_param {
        scale: 0.00390625
      }
      data_param {
        source: "examples/mnist/mnist_train_lmdb"
        batch_size: 64
        backend: LMDB
      }
    }
    layer {
      name: "mnist"
      type: "Data"
      top: "data"
      top: "label"
      include {
        phase: TEST
      }
      transform_param {
        scale: 0.00390625
      }
      data_param {
        source: "examples/mnist/mnist_test_lmdb"
        batch_size: 100
        backend: LMDB
      }
    }
    layer {
      name: "conv1"
      type: "Convolution"
      bottom: "data"
      top: "conv1"
      param {
        lr_mult: 1
      }
      param {
        lr_mult: 2
      }
      convolution_param {
        num_output: 20
        kernel_size: 5
        stride: 1
        weight_filler {
          type: "xavier"
        }
        bias_filler {
          type: "constant"
        }
      }
    }
    layer {
      name: "pool1"
      type: "Pooling"
      bottom: "conv1"
      top: "pool1"
      pooling_param {
        pool: MAX
        kernel_size: 2
        stride: 2
      }
    }
    layer {
      name: "conv2"
      type: "Convolution"
      bottom: "pool1"
      top: "conv2"
      param {
        lr_mult: 1
      }
      param {
        lr_mult: 2
      }
      convolution_param {
        num_output: 50
        kernel_size: 5
        stride: 1
        weight_filler {
          type: "xavier"
        }
        bias_filler {
          type: "constant"
        }
      }
    }
    layer {
      name: "pool2"
      type: "Pooling"
      bottom: "conv2"
      top: "pool2"
      pooling_param {
        pool: MAX
        kernel_size: 2
        stride: 2
      }
    }
    layer {
      name: "ip1"
      type: "InnerProduct"
      bottom: "pool2"
      top: "ip1"
      param {
        lr_mult: 1
      }
      param {
        lr_mult: 2
      }
      inner_product_param {
        num_output: 500
        weight_filler {
          type: "xavier"
        }
        bias_filler {
          type: "constant"
        }
      }
    }
    layer {
      name: "relu1"
      type: "ReLU"
      bottom: "ip1"
      top: "ip1"
    }
    layer {
      name: "ip2"
      type: "InnerProduct"
      bottom: "ip1"
      top: "ip2"
      param {
        lr_mult: 1
      }
      param {
        lr_mult: 2
      }
      inner_product_param {
        num_output: 10
        weight_filler {
          type: "xavier"
        }
        bias_filler {
          type: "constant"
        }
      }
    }
    layer {
      name: "accuracy"
      type: "Accuracy"
      bottom: "ip2"
      bottom: "label"
      top: "accuracy"
      include {
        phase: TEST
      }
    }
    layer {
      name: "loss"
      type: "SoftmaxWithLoss"
      bottom: "ip2"
      bottom: "label"
      top: "loss"
    }
    )";

    auto model = create_net_from_json(json, shape3d(28, 28, 1));

    // conv->pool->conv->pool->fc->relu->fc->softmax
    ASSERT_EQ(model->depth(), 8);

    // conv1 28x28x1 -> 24x24x20
    EXPECT_EQ((*model)[0]->in_shape()[0], shape3d(28, 28, 1));   // in: 28x28x1
    EXPECT_EQ((*model)[0]->in_shape()[1], shape3d(5, 5, 20));    // weight: 5x5x20
    EXPECT_EQ((*model)[0]->in_shape()[2], shape3d(1, 1, 20));    // bias: 1x1x20
    EXPECT_EQ((*model)[0]->out_shape()[0], shape3d(24, 24, 20)); // out:24x24x20
    EXPECT_EQ((*model)[0]->layer_type(), "conv");

    // pool1 24x24x20 -> 12x12x20
    EXPECT_EQ((*model)[1]->in_shape()[0], shape3d(24, 24, 20));
    EXPECT_EQ((*model)[1]->out_shape()[0], shape3d(12, 12, 20));
    EXPECT_EQ((*model)[1]->layer_type(), "max-pool");

    // conv2 12x12x20 -> 8x8x50
    EXPECT_EQ((*model)[2]->in_shape()[0], shape3d(12, 12, 20));
    EXPECT_EQ((*model)[2]->in_shape()[1], shape3d(5, 5, 1000));
    EXPECT_EQ((*model)[2]->in_shape()[2], shape3d(1, 1, 50));
    EXPECT_EQ((*model)[2]->out_shape()[0], shape3d(8, 8, 50));
    EXPECT_EQ((*model)[2]->layer_type(), "conv");

    // pool2 8x8x50 -> 4x4x50
    EXPECT_EQ((*model)[3]->in_shape()[0], shape3d(8, 8, 50));
    EXPECT_EQ((*model)[3]->out_shape()[0], shape3d(4, 4, 50));
    EXPECT_EQ((*model)[3]->layer_type(), "max-pool");

    // fc
    EXPECT_EQ((*model)[4]->in_shape()[0], shape3d(4 * 4 * 50, 1, 1));
    EXPECT_EQ((*model)[4]->out_shape()[0], shape3d(500, 1, 1));
    EXPECT_EQ((*model)[4]->layer_type(), "fully-connected");

    // relu
    EXPECT_EQ((*model)[5]->in_shape()[0], shape3d(500, 1, 1));
    EXPECT_EQ((*model)[5]->out_shape()[0], shape3d(500, 1, 1));
    EXPECT_EQ((*model)[5]->layer_type(), "linear");

    // fc
    EXPECT_EQ((*model)[6]->in_shape()[0], shape3d(500, 1, 1));
    EXPECT_EQ((*model)[6]->out_shape()[0], shape3d(10, 1, 1));
    EXPECT_EQ((*model)[6]->layer_type(), "fully-connected");

    // softmax
    EXPECT_EQ((*model)[7]->in_shape()[0], shape3d(10, 1, 1));
    EXPECT_EQ((*model)[7]->out_shape()[0], shape3d(10, 1, 1));
    EXPECT_EQ((*model)[7]->layer_type(), "linear");
}

TEST(caffe_converter, conv2) {

}

} // namespace tiny-dnn
