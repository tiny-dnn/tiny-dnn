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
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#include "gtest/gtest.h"
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn::activation;

#ifndef CNN_NO_SERIALIZATION
#include "test_serialization.h"
#endif
#include "test_network.h"
#include "test_average_pooling_layer.h"
// TODO(yida): fix broken test
//#include "test_average_unpooling_layer.h"
#include "test_dropout_layer.h"
#include "test_max_pooling_layer.h"
#include "test_fully_connected_layer.h"
#include "test_deconvolutional_layer.h"
#include "test_convolutional_layer.h"
#include "test_target_cost.h"
#include "test_large_thread_count.h"
#include "test_lrn_layer.h"
#include "test_batch_norm_layer.h"
#include "test_nodes.h"
#include "test_core.h"
#include "test_models.h"
#include "test_slice_layer.h"
#include "test_concat_layer.h"
#include "test_power_layer.h"
#include "test_quantization.h"
#include "test_quantized_convolutional_layer.h"
#include "test_quantized_deconvolutional_layer.h"
#ifdef CNN_USE_GEMMLOWP
#include "test_quantized_fully_connected_layer.h"
#endif

#ifdef CNN_USE_CAFFE_CONVERTER
#include "test_caffe_converter.h"
#endif

#include "test_tensor.h"
#include "test_image.h"

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
