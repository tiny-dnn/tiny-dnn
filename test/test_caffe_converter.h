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


TEST(caffe_converter, lenet_v1) {
    /*
     * loading caffe's old version prototxt
     *
     */ 
    std::string json = R"(
    name: "LeNet"
    input: "data"
    input_dim: 64
    input_dim: 1
    input_dim: 28
    input_dim: 28
    layers {
      name: "conv1"
      type: CONVOLUTION
      bottom: "data"
      top: "conv1"
      blobs_lr: 1
      blobs_lr: 2
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
    layers {
      name: "pool1"
      type: POOLING
      bottom: "conv1"
      top: "pool1"
      pooling_param {
        pool: MAX
        kernel_size: 2
        stride: 2
      }
    }
    layers {
      name: "conv2"
      type: CONVOLUTION
      bottom: "pool1"
      top: "conv2"
      blobs_lr: 1
      blobs_lr: 2
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
    layers {
      name: "pool2"
      type: POOLING
      bottom: "conv2"
      top: "pool2"
      pooling_param {
        pool: MAX
        kernel_size: 2
        stride: 2
      }
    }
    layers {
      name: "ip1"
      type: INNER_PRODUCT
      bottom: "pool2"
      top: "ip1"
      blobs_lr: 1
      blobs_lr: 2
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
    layers {
      name: "relu1"
      type: RELU
      bottom: "ip1"
      top: "ip1"
    }
    layers {
      name: "ip2"
      type: INNER_PRODUCT
      bottom: "ip1"
      top: "ip2"
      blobs_lr: 1
      blobs_lr: 2
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
    layers {
      name: "prob"
      type: SOFTMAX
      bottom: "ip2"
      top: "prob"
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

TEST(caffe_converter, dropout) {
    std::string json = R"(
    name: "DropoutNet"
    input: "data"
    input_shape {
      dim: 1
      dim: 1
      dim: 40
      dim: 24
    }
    layer {
      name: "dropout"
      type: "Dropout"
      bottom: "data"
      top: "dropout"
      dropout_param {
        dropout_ratio: 0.3
      }
    }
    )";

    auto model = create_net_from_json(json);

    ASSERT_EQ(model->depth(), 1);

    // tiny-dnn dropout doesn't hold spatial shape of input
    EXPECT_EQ((*model)[0]->in_shape()[0], shape3d(24*40, 1, 1));
    EXPECT_EQ((*model)[0]->out_shape()[0], shape3d(24*40, 1, 1));
    EXPECT_FLOAT_EQ(model->at<dropout_layer>(0).dropout_rate(), 0.3f);
}


TEST(caffe_converter, conv_with_stride) {
    std::string json = R"(
    name: "DropoutNet"
    input: "data"
    input_shape {
      dim: 1
      dim: 2
      dim: 40
      dim: 24
    }
    layer {
      name: "conv1"
      type: "Convolution"
      bottom: "data"
      top: "conv1"
      convolution_param {
        num_output: 3
        pad: 2
        kernel_size: 5
        stride_w: 3
        stride_h: 2
        bias_term: false
      }
    }
    )";

    auto model = create_net_from_json(json);

    ASSERT_EQ(model->depth(), 1);

    EXPECT_EQ((*model)[0]->in_shape()[0], shape3d(24, 40, 2));
    EXPECT_EQ((*model)[0]->in_shape().size(), 2); // doesn't have bias
    EXPECT_EQ((*model)[0]->out_shape()[0], shape3d(8, 20, 3));
}

TEST(caffe_converter, batchnorm) {
    std::string json = R"(
    name: "BNNet"
    input: "data"
    input_shape {
      dim: 1
      dim: 1
      dim: 40
      dim: 24
    }
    layer {
      name: "bn"
      type: "BatchNorm"
      bottom: "data"
      top: "normalized"
      batch_norm_param {
        moving_average_fraction: 0.8
        eps: 1e-3
      }
    }
    )";

    auto model = create_net_from_json(json);

    ASSERT_EQ(model->depth(), 1);

    // tiny-dnn bn doesn't hold spatial shape of input
    EXPECT_EQ((*model)[0]->in_shape()[0], shape3d(24*40, 1, 1));
    EXPECT_EQ((*model)[0]->out_shape()[0], shape3d(24*40, 1, 1));
    EXPECT_EQ((*model)[0]->layer_type(), "batch-norm");
    EXPECT_FLOAT_EQ(model->at<batch_normalization_layer>(0).epsilon(), 1e-3f);
    EXPECT_FLOAT_EQ(model->at<batch_normalization_layer>(0).momentum(), 0.8f);
}

TEST(caffe_converter, ave_pool) {
    std::string json = R"(
    name: "PoolNet"
    input: "data"
    input_shape {
      dim: 1
      dim: 2
      dim: 40
      dim: 24
    }
    layer {
      name: "pool"
      type: "Pooling"
      bottom: "data"
      top: "pooled"
      pooling_param {
        pool: AVE
        kernel_size: 2
        stride: 2
      }
    }
    )";

    auto model = create_net_from_json(json);

    ASSERT_EQ(model->depth(), 1);
    EXPECT_EQ((*model)[0]->in_shape()[0], shape3d(24, 40, 2));
    EXPECT_EQ((*model)[0]->out_shape()[0], shape3d(12, 20, 2));
    EXPECT_EQ((*model)[0]->layer_type(), "ave-pool");
}

TEST(caffe_converter, deconv) {
    std::string json = R"(
    name: "DeconvNet"
    input: "data"
    input_shape {
      dim: 1
      dim: 4
      dim: 40
      dim: 24
    }
    layer {
      name: "conv"
      type: "Deconvolution"
      bottom: "data"
      top: "out"
      convolution_param {
        num_output: 10
        pad: 1
        kernel_size: 3
        group: 2
      }
    }
    )";

    auto model = create_net_from_json(json);

    ASSERT_EQ(model->depth(), 1);
    EXPECT_EQ((*model)[0]->in_shape()[0], shape3d(24, 40, 4));
    EXPECT_EQ((*model)[0]->out_shape()[0], shape3d(24, 40, 10));
}

TEST(caffe_converter, lrn) {
    std::string json = R"(
    name: "LRNNet"
    input: "data"
    input_shape {
      dim: 1
      dim: 5
      dim: 40
      dim: 24
    }
    layer {
      name: "lrn"
      type: "LRN"
      bottom: "data"
      top: "normalized"
      lrn_param {
        norm_region: WITHIN_CHANNEL
        alpha: 1.5
        beta: 0.8
        local_size: 3
      }
    }
    )";

    auto model = create_net_from_json(json);

    ASSERT_EQ(model->depth(), 1);

    EXPECT_EQ((*model)[0]->in_shape()[0], shape3d(24, 40, 5));
    EXPECT_EQ((*model)[0]->out_shape()[0], shape3d(24, 40, 5));
}

TEST(caffe_converter, sigmoid) {
    std::string json = R"(
    name: "SigmoidNet"
    input: "data"
    input_shape {
      dim: 1
      dim: 1
      dim: 5
      dim: 1
    }
    layer {
      name: "activation"
      type: "Sigmoid"
      bottom: "data"
      top: "out"
    }
    )";

    auto model = create_net_from_json(json);

    ASSERT_EQ(model->depth(), 1);

    EXPECT_EQ((*model)[0]->in_shape()[0], shape3d(5, 1, 1));
    EXPECT_EQ((*model)[0]->out_shape()[0], shape3d(5, 1, 1));

    vec_t in = { 0.0f, 0.1f, 0.5f, 0.9f, 1.0f };

    auto ret = model->predict(in);
    sigmoid a;

    EXPECT_EQ(ret[0], a.f(in, 0));
    EXPECT_EQ(ret[1], a.f(in, 1));
    EXPECT_EQ(ret[2], a.f(in, 2));
    EXPECT_EQ(ret[3], a.f(in, 3));
    EXPECT_EQ(ret[4], a.f(in, 4));
}

TEST(caffe_converter, tanh) {
    std::string json = R"(
    name: "SigmoidNet"
    input: "data"
    input_shape {
      dim: 1
      dim: 1
      dim: 1
      dim: 5
    }
    layer {
      name: "activation"
      type: "TanH"
      bottom: "data"
      top: "out"
    }
    )";

    auto model = create_net_from_json(json);

    ASSERT_EQ(model->depth(), 1);

    EXPECT_EQ((*model)[0]->in_shape()[0], shape3d(5, 1, 1));
    EXPECT_EQ((*model)[0]->out_shape()[0], shape3d(5, 1, 1));

    vec_t in = { -1.0f, -0.1f, 0.0f, 0.1f, 1.0f };

    auto ret = model->predict(in);
    tan_h a;

    EXPECT_EQ(ret[0], a.f(in, 0));
    EXPECT_EQ(ret[1], a.f(in, 1));
    EXPECT_EQ(ret[2], a.f(in, 2));
    EXPECT_EQ(ret[3], a.f(in, 3));
    EXPECT_EQ(ret[4], a.f(in, 4));
}

TEST(caffe_converter, power) {
    std::string json = R"(
    name: "Power"
    input: "data"
    input_shape {
      dim: 1
      dim: 1
      dim: 1
      dim: 1
    }
    layer {
      name: "pow"
      type: "Power"
      bottom: "data"
      top: "out"
      power_param {
        power: 0.5
        scale: 2.0
      }
    }
    )";

    auto model = create_net_from_json(json);

    ASSERT_EQ(model->depth(), 1);

    EXPECT_EQ((*model)[0]->in_shape()[0], shape3d(1, 1, 1));
    EXPECT_EQ((*model)[0]->out_shape()[0], shape3d(1, 1, 1));
    EXPECT_FLOAT_EQ((*model).at<power_layer>(0).factor(), 0.5f);
    EXPECT_FLOAT_EQ((*model).at<power_layer>(0).scale(), 2.0f);
}

TEST(caffe_converter, conv_with_weights) {
    std::string json = R"(
    name: "ConvNet"
    input: "data"
    input_shape {
      dim: 1
      dim: 1
      dim: 3
      dim: 3
    }
    layer {
      name: "conv"
      type: "Convolution"
      bottom: "data"
      top: "out"
      convolution_param {
        num_output: 1
        kernel_size: 3
        stride: 1
      }
      blobs {
        data: 0
        data: 1
        data: 2
        data: 3
        data: 4
        data: 5
        data: 6
        data: 7
        data: 8
        shape {
          dim: 1
          dim: 1
          dim: 3
          dim: 3
        }
      }
      blobs {
        data: 9
        shape {
          dim: 1
          dim: 1
          dim: 1
          dim: 1
        }
      }
    }
    )";

    auto model = create_net_from_json(json);

    ASSERT_EQ(model->depth(), 1);

    const vec_t* W = (*model)[0]->weights()[0];
    const vec_t* b = (*model)[0]->weights()[1];

    EXPECT_EQ(W->size(), 9);
    for (int i = 0; i < 9; i++) {
        EXPECT_FLOAT_EQ(W->at(i), (float_t)i);
    }
    EXPECT_EQ(b->size(), 1);
    EXPECT_FLOAT_EQ(b->at(0), 9.0f);
}



TEST(caffe_converter, fully_with_weights) {
    std::string json = R"(
    name: "FcNet"
    input: "data"
    input_shape {
      dim: 1
      dim: 1
      dim: 2
      dim: 2
    }
    layer {
      name: "fc"
      type: "InnerProduct"
      bottom: "data"
      top: "out"
      inner_product_param {
        num_output: 2
        bias_term: true
      }
      blobs {
        data: 0
        data: 1
        data: 2
        data: 3
        data: 4
        data: 5
        data: 6
        data: 7
        shape {
          dim: 1
          dim: 1
          dim: 2
          dim: 4
        }
      }
      blobs {
        data: 8
        data: 9
        shape {
          dim: 1
          dim: 1
          dim: 1
          dim: 2
        }
      }
    }
    )";

    auto model = create_net_from_json(json);

    ASSERT_EQ(model->depth(), 1);
    /*
     caffe:
     0 1 2 3
     4 5 6 7

     tiny-dnn:
     0 4
     1 5
     2 6
     3 7 
    */

    const vec_t* W = (*model)[0]->weights()[0];
    const vec_t* b = (*model)[0]->weights()[1];

    EXPECT_EQ(W->size(), 8);
    EXPECT_FLOAT_EQ(W->at(0), 0.0f);
    EXPECT_FLOAT_EQ(W->at(1), 4.0f);
    EXPECT_FLOAT_EQ(W->at(2), 1.0f);
    EXPECT_FLOAT_EQ(W->at(3), 5.0f);
    EXPECT_FLOAT_EQ(W->at(4), 2.0f);
    EXPECT_FLOAT_EQ(W->at(5), 6.0f);
    EXPECT_FLOAT_EQ(W->at(6), 3.0f);
    EXPECT_FLOAT_EQ(W->at(7), 7.0f);

    EXPECT_EQ(b->size(), 2);
    EXPECT_FLOAT_EQ(b->at(0), 8.0f);
    EXPECT_FLOAT_EQ(b->at(1), 9.0f);
}

} // namespace tiny-dnn
