/*
    Copyright (c) 2016, Taiga Nomi, Edgar Riba
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

using namespace tiny_dnn;

namespace tiny_dnn {

TEST(tensor, shape) {
    Tensor<float_t,1,2,2,2> tensor;

    EXPECT_EQ(tensor.shape()[0], cnn_size_t(1));
    EXPECT_EQ(tensor.shape()[1], cnn_size_t(2));
    EXPECT_EQ(tensor.shape()[2], cnn_size_t(2));
    EXPECT_EQ(tensor.shape()[3], cnn_size_t(2));
}

TEST(tensor, access_data) {
    Tensor<float_t,1,2,2,2> tensor;

    float_t* begin_ptr = tensor.ptr<float_t>(0,0,0,0);
    float_t* end1_ptr  = tensor.ptr<float_t>(0,1,1,0);
    float_t* end2_ptr  = tensor.ptr<float_t>(0,1,1,1);

    // set tensor data
    
    // channel #1
    for (float_t* i = begin_ptr; i != end1_ptr + 1; i++) {
        *i = float_t(1);
    }

    // channel #2
    for (float_t* i = end1_ptr + 1; i != end2_ptr + 1; i++) {
        *i = float_t(2);
    }

    // check data
    
    for (cnn_size_t i = 0; i < 2; ++i) {
        for (cnn_size_t j = 0; j < 2; ++j) {
            for (cnn_size_t k = 0; k < 1; ++k) {
                if (k == 0) {
                    EXPECT_EQ(tensor.at<float_t>(0,i,j,k), cnn_size_t(1));
                } else {
                    EXPECT_EQ(tensor.at<float_t>(0,i,j,k), cnn_size_t(2));
                }
            }
        }
    }
}

} // namespace tiny-dnn
