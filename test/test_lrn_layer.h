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

TEST(lrn, cross) {
    lrn_layer<identity> lrn(1, 1, 3, 4, /*alpha=*/1.5, /*beta=*/2.0, norm_region::across_channels);

    tiny_dnn::float_t in[4] = { -1.0, 3.0, 2.0, 5.0 };
    tiny_dnn::float_t expected[4] =
    {
        -1.0f/36.0f,    // -1.0 / (1+0.5*(1*1+3*3))^2
        3.0f/64.0f,     //  3.0 / (1+0.5*(1*1+3*3+2*2))^2
        2.0f/400.0f,    //  2.0 / (1+0.5*(3*3+2*2+5*5))^2
        5.0f/15.5f/15.5f // 5.0 / (1+0.5*(2*2+5*5))^2
    };

    auto out = lrn.forward({ {vec_t(in, in + 4)} })[0][0];

    EXPECT_NEAR(expected[0], out[0], epsilon<float_t>());
    EXPECT_NEAR(expected[1], out[1], epsilon<float_t>());
    EXPECT_NEAR(expected[2], out[2], epsilon<float_t>());
    EXPECT_NEAR(expected[3], out[3], epsilon<float_t>());
}

TEST(lrn, read_write) {
    lrn_layer<identity> l1(10, 10, 3, 4, 1.5f, 2.0f, norm_region::across_channels);
    lrn_layer<identity> l2(10, 10, 3, 4, 1.5f, 2.0f, norm_region::across_channels);

    l1.init_weight();
    l2.init_weight();

    serialization_test(l1, l2);
}

} // namespace tiny-dnn
