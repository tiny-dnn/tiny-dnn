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
  size_t in_size(state.range(0)), out_size(state.range(1));

  Tensor<> in_data({1, in_size}), out_data({1, out_size});
  Tensor<> weights({in_size * out_size}), bias({out_size});

  uniform_rand(in_data.host_begin(), in_data.host_end(), -1, 1);
  uniform_rand(weights.host_begin(), weights.host_end(), -1, 1);
  uniform_rand(bias.host_begin(), bias.host_end(), -1, 1);
  uniform_rand(out_data.host_begin(), out_data.host_end(), -1, 1);

  while (state.KeepRunning()) {
    kernels::fully_connected_op_internal(in_data, weights, bias, out_data,
                                         false);
  }
}

BENCHMARK(bm_fully_connected_internal)
  ->Args({16, 4})
  ->Args({64, 4})
  ->Args({256, 4})
  ->Args({1024, 4})
  ->Args({1024, 16})
  ->Args({1024, 64})
  ->Args({1024, 256})
  ->Args({1024, 1024});

}  // namespace benchmarks
}  // namespace tiny_dnn
