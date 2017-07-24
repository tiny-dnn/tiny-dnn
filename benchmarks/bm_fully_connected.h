/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "benchmark/benchmark.h"
#include "tiny_dnn/tiny_dnn.h"

namespace bm = benchmark;

namespace tiny_dnn {
namespace benchmarks {

void bm_fully_connected_internal(bm::State& state) {
  Tensor<float_t> in_data({1, 5}), out_data({1, 3}), weights({15}), bias({3});

  uniform_rand(in_data.host_begin(), in_data.host_end(), -1, 1);
  uniform_rand(weights.host_begin(), weights.host_end(), -1, 1);
  uniform_rand(bias.host_begin(), bias.host_end(), -1, 1);
  uniform_rand(out_data.host_begin(), out_data.host_end(), -1, 1);

  while (state.KeepRunning()) {
    kernels::fully_connected_op_internal(in_data, weights, bias, out_data,
                                         false);
  }
}

BENCHMARK(bm_fully_connected_internal);

}  // namespace benchmarks
}  // namespace tiny_dnn
