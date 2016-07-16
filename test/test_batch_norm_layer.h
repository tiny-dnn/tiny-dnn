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
#include "picotest/picotest.h"
#include "testhelper.h"
#include "tiny_cnn/tiny_cnn.h"

namespace tiny_cnn {

TEST(batchnorm, gradient_check) {
    network<sequential> nn;
    nn << convolutional_layer<relu>(5, 5, 3, 3, 3, padding::same)
        << batch_normalization_layer(3, 5 * 5)
        << fully_connected_layer<tan_h>(3 * 5 * 5, 10);
    
    nn.at<batch_normalization_layer>(1).update_immidiately(true);

    const auto test_data = generate_gradient_check_data(nn.in_data_size(), 2);
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second, 1e-4, GRAD_CHECK_ALL));

}

TEST(batchnorm, forward) {
    batch_normalization_layer bn(/*channel=*/3, /*spatial-size=*/4);

    /*
          mean   var
    ch0:  0.0    0.0
    ch1: -1.0    6.0
    ch2:  2.875 10.696
    */
    tensor_t in = {
      {
         0.0f,  0.0f,  0.0f,  0.0f, // ch-0 of data#0
        -4.0f,  0.0f, -1.0f,  2.0f, // ch-1 of data#0
         1.0f,  0.0f,  1.0f,  3.0f, // ch-2 of data#0
      }, {
         0.0f,  0.0f,  0.0f,  0.0f,  // ch-0 of data#1  
         2.0f,  0.0f, -4.0f, -3.0f,  // ch-1 of data#1
         2.0f,  5.0f,  1.0f, 10.0f   // ch-2 of data#1
      }
    };

    /* y = (x - mean) ./ sqrt(variance + eps) */
    tensor_t expect = {
        {
            0.0f,    0.0f,    0.0f,   0.0f,   // ch-0 of data#0
           -1.225f,  0.408f,  0.0f,   1.225f, // ch-1 of data#0
           -0.573f, -0.879f, -0.573f, 0.038f, // ch-2 of data#0
        },{
            0.0f,   0.0f,    0.0f,    0.0f,  // ch-0 of data#1  
            1.225f, 0.408f, -1.225f, -0.816f,  // ch-1 of data#1
           -0.268f, 0.650f, -0.573f,  2.179f   // ch-2 of data#1
        }
    };

    auto result = bn.forward({ in });

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3 * 4; j++) {
            EXPECT_NEAR(expect[i][j], result[0][i][j], 1e-3);
        }
    }

    bn.post_update();

    // confirming that calculating the moving average doesn't affect the result
    // while we feed the same data
    result = bn.forward({ in });
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3 * 4; j++) {
            EXPECT_NEAR(expect[i][j], result[0][i][j], 1e-3);
        }
    }
}

TEST(batchnorm, read_write) {

}

} // namespace tiny-cnn
