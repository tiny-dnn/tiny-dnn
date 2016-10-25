// Copyright (c) 2013-2016, Taiga Nomi. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#pragma once
#include "picotest/picotest.h"
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
