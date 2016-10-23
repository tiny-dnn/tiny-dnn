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

TEST(batchnorm, gradient_check) {
    int num = 4;
    int spatial_dim = 4;
    int channels = 2;
    batch_normalization_layer bn(spatial_dim, channels);

    /* following values are extracted from caffe */
    /* confirming that batch-norm layer is compatible with caffe's bn */

    float_t top_diff[] = {
        0.554228544,
        -0.823364496,
        -0.103415221,
        0.669684947,
        0.142640188,
        -0.171076611,
        0.292261183,
        -0.067076027,
        -0.00277741,
        0.058186941,
        0.046050139,
        -0.006042562,
        -0.004771964,
        0.025202896,
        -0.062344212,
        0.030099955,
        -0.023314178,
        -0.030725746,
        0.070954606,
        0.055909708,
        -0.019887319,
        0.076775789,
        0.014769247,
        -0.025637595,
        0.004412052,
        -0.013895055,
        -0.001271803,
        3.15E-05,
        -0.013110356,
        0.008091689,
        -0.005485342,
        0.007250476
    };


    float_t top_data[] = {
        -0.430924207,
        -2.23937607,
        1.7876749,
        1.41079676,
        0.578419685,
        0.662835836,
        -2.1911881,
        -0.002405337,
        1.49315703,
        -0.836038888,
        0.006807627,
        -0.012308626,
        0.424309582,
        -0.56077528,
        0.095194906,
        0.34416762,
        -0.755284429,
        -1.02720368,
        0.802836478,
        -0.06101859,
        2.17714667,
        -0.994640052,
        -0.497716337,
        0.397495717,
        -0.545207798,
        0.320612997,
        -0.016919944,
        0.102396645,
        0.551594019,
        -1.44724381,
        -0.530790627,
        0.993595243
    };

    float_t stddev[] = {
        5.13347721,
        6.15658283
    };

    float_t expected_gradients[] = {
        0.115063809,
        -0.100263298,
        -0.078099057,
        0.083551511,
        0.025724376,
        -0.024521608,
        0.026721604,
        -0.01322682,
        -0.049858954,
        0.030313857,
        0.003235561,
        -0.006351554,
        0.000483762,
        -0.002936667,
        -0.011636967,
        0.00547356,
        0.012069567,
        0.018599048,
        -0.015254314,
        0.007145011,
        0.01277818,
        0.001789367,
        -0.004100761,
        -0.003131026,
        0.011310737,
        -0.017643189,
        -0.005286998,
        -0.008531732,
        0.000200434,
        -0.013175356,
        -0.007668978,
        0.007226899
    };

    tensor_t outd, ing, outg;
    std::vector<tensor_t*> in_data, out_data, in_grad, out_grad;

    for (int i = 0; i < num; i++) {
        int first = i*spatial_dim*channels;
        int last = first + spatial_dim*channels;

        ing.push_back(vec_t(spatial_dim*channels));
        outg.push_back(vec_t(top_diff + first, top_diff + last));
        outd.push_back(vec_t(top_data + first, top_data + last));
    }
    in_grad.push_back(&ing);
    out_grad.push_back(&outg);
    out_data.push_back(&outd);

    bn.set_context(net_phase::train);
    bn.set_stddev(vec_t(stddev, stddev + 2));
    bn.back_propagation(in_data, out_data, out_grad, in_grad);

    for (int i = 0; i < num; i++) {
        for (int j = 0; j < spatial_dim*channels; j++) {
            EXPECT_NEAR(expected_gradients[i*spatial_dim*channels + j], (*in_grad[0])[i][j], 1e-4);
        }
    }
}

TEST(batchnorm, forward) {
    batch_normalization_layer bn(/*spatial-size=*/4, /*channel=*/3);

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
    batch_normalization_layer l1(100, 100);
    batch_normalization_layer l2(100, 100);

    l1.setup(true);
    l2.setup(true);

    serialization_test(l1, l2);
}

} // namespace tiny-dnn
