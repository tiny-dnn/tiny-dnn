/*
    Copyright (c)
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
#include <vector>
#include "gtest/gtest.h"
#include "test/testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

namespace tiny_dnn {

TEST(test_large_thread_count, test_large_thread_count) {
    network<sequential> net;
    net << fully_connected_layer<tan_h>(1, 2);
    adagrad optimizer;

    std::vector<vec_t> data;
    std::vector<label_t> labels;

    const size_t tnum = 100;

    for (size_t i = 0; i < tnum; i++) {
        bool in = bernoulli(0.5);
        bool label = bernoulli(0.5);

        data.push_back({ static_cast<float_t>(in) });
        labels.push_back(label ? 1 : 0);
    }

    const int n_threads = 200;

    // test different batch sizes
    net.train<mse>(optimizer, data, labels, 1, 1, nop, nop, true, n_threads);
    net.train<mse>(optimizer, data, labels, 100, 1, nop, nop, true, n_threads);
    net.train<mse>(optimizer, data, labels, 200, 1, nop, nop, true, n_threads);
    net.train<mse>(optimizer, data, labels, 300, 1, nop, nop, true, n_threads);
}

}  // namespace tiny_dnn
