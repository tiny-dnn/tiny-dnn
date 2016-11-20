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

TEST(power, forward) {
    power_layer pw(shape3d(3,2,1), 2.0, 1.5);

    tensor_t in = {
        { 0,1,2,3,4,5 },
        { -5,-4,-3,-2,-1,0 },
    };

    tensor_t out_expected = {
        { 0*0*1.5,1*1*1.5,2*2*1.5,3*3*1.5,4*4*1.5,5*5*1.5 },
        { 5*5*1.5,4*4*1.5,3*3*1.5,2*2*1.5,1*1*1.5,0*0*1.5 }
    };

    auto out = pw.forward({in});

    for (serial_size_t i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(out_expected[0][i], out[0][0][i]);
        EXPECT_FLOAT_EQ(out_expected[1][i], out[0][1][i]);
    }
}


TEST(power, gradient_check) {
    network<sequential> nn;

    nn << fully_connected_layer<tan_h>(10, 20)
       << power_layer(shape3d(20, 1, 1), 3.0, 1.5)
       << fully_connected_layer<tan_h>(20, 10);

    const auto test_data = generate_gradient_check_data(nn.in_data_size());
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second, epsilon<float_t>(), GRAD_CHECK_ALL));
}

} // namespace tiny-dnn
