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

TEST(slice, forward_data) {
    slice_layer sl(shape3d(3,2,1), slice_type::slice_samples, 3);

    tensor_t in = {
        { 0,1,2,3,4,5 },
        { 6,7,8,9,10,11 },
        { 12,13,14,15,16,17 },
        { 18,19,20,21,22,23 }
    };

    tensor_t out0_expected = {
        { 0,1,2,3,4,5 }
    };

    tensor_t out1_expected = {
        { 6,7,8,9,10,11 }
    };

    tensor_t out2_expected = {
        { 12, 13, 14, 15, 16, 17 },
        { 18, 19, 20, 21, 22, 23 }
    };

    auto out = sl.forward({in});

    for (serial_size_t i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(out0_expected[0][i], out[0][0][i]);
        EXPECT_FLOAT_EQ(out1_expected[0][i], out[1][0][i]);
        EXPECT_FLOAT_EQ(out2_expected[0][i], out[2][0][i]);
        EXPECT_FLOAT_EQ(out2_expected[1][i], out[2][1][i]);
    }

    out = sl.backward({out0_expected,out1_expected,out2_expected});

    for (serial_size_t i = 0; i < 4; i++) {
        for (serial_size_t j = 0; j < 6; j++) {
            EXPECT_FLOAT_EQ(in[i][j], out[0][i][j]);
        }
    }
}

TEST(slice, forward_channels) {
    slice_layer sl(shape3d(1, 2, 3), slice_type::slice_channels, 3);

    tensor_t in = {
        { 0, 1, 2, 3, 4, 5 },
        { 6, 7, 8, 9, 10, 11 },
        { 12, 13, 14, 15, 16, 17 },
        { 18, 19, 20, 21, 22, 23 }
    };

    tensor_t out0_expected = {
        { 0, 1 },
        { 6, 7 },
        { 12, 13 },
        { 18, 19 }
    };

    tensor_t out1_expected = {
        {  2, 3 },
        {  8, 9 },
        { 14, 15 },
        { 20, 21 }
    };

    tensor_t out2_expected = {
        { 4, 5 },
        { 10, 11 },
        { 16, 17 },
        { 22, 23 }
    };

    auto out = sl.forward({ in });

    for (serial_size_t i = 0; i < 4; i++) {
        for (serial_size_t j = 0; j < 2; j++) {
            EXPECT_FLOAT_EQ(out0_expected[i][j], out[0][i][j]);
            EXPECT_FLOAT_EQ(out1_expected[i][j], out[1][i][j]);
            EXPECT_FLOAT_EQ(out2_expected[i][j], out[2][i][j]);
        }
    }

    out = sl.backward({out0_expected, out1_expected, out2_expected});

    for (serial_size_t i = 0; i < 4; i++) {
        for (serial_size_t j = 0; j < 6; j++) {
            EXPECT_FLOAT_EQ(in[i][j], out[0][i][j]);
        }
    }
}

} // namespace tiny-dnn
