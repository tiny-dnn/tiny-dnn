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
#include <deque>

namespace tiny_dnn {

TEST(dropout, randomized) {
    int num_units = 10000;
    tiny_dnn::float_t dropout_rate = 0.1f;
    dropout_layer l(num_units, dropout_rate, net_phase::train);
    vec_t v(num_units, 1.0);

    l.forward({ {v} });
    const auto mask1 = l.get_mask(0);

    l.forward({ {v} });
    const auto mask2 = l.get_mask(0);

    // mask should change for each fprop
    EXPECT_TRUE(is_different_container(mask1, mask2));

    // dropout-rate should be around 0.1
    double margin_factor = 0.9;
    int64_t num_on1 = std::count(mask1.begin(), mask1.end(), 1);
    int64_t num_on2 = std::count(mask2.begin(), mask2.end(), 1);

    EXPECT_LE(num_units * dropout_rate * margin_factor, num_on1);
    EXPECT_GE(num_units * dropout_rate / margin_factor, num_on1);
    EXPECT_LE(num_units * dropout_rate * margin_factor, num_on2);
    EXPECT_GE(num_units * dropout_rate / margin_factor, num_on2);
}

TEST(dropout, read_write) {
    dropout_layer l1(1024, 0.5, net_phase::test);
    dropout_layer l2(1024, 0.5, net_phase::test);

    l1.init_weight();
    l2.init_weight();

    serialization_test(l1, l2);
}

} // namespace tiny-dnn
