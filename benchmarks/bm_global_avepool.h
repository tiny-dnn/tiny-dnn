/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <tuple>

#include "benchmark/benchmark.h"
#include "tiny_dnn/tiny_dnn.h"

namespace tiny_dnn {
namespace benchmarks {

std::tuple<tensor_t, tensor_t, core::global_avepool_params>
get_bm_global_avepool_data() {
  static constexpr size_t SIZE_IN = 100;
  vec_t input_one(SIZE_IN * SIZE_IN * SIZE_IN, SIZE_IN * SIZE_IN);
  vec_t output_one(SIZE_IN, 0);
  tensor_t input_data(1, input_one);
  tensor_t output_data(1, output_one);
  core::global_avepool_params params;
  params.in  = {SIZE_IN, SIZE_IN, SIZE_IN};
  params.out = {SIZE_IN, 1, 1};
  return std::make_tuple(input_data, output_data, params);
}

void bm_global_avepool_forward_internal(benchmark::State& state) {
  tensor_t input_data, output_data;
  core::global_avepool_params params;
  std::tie(input_data, output_data, params) = get_bm_global_avepool_data();

  while (state.KeepRunning()) {
    kernels::global_avepool_op_internal(input_data, output_data, params, true);
  }
}

#ifdef CNN_USE_AVX
void bm_global_avepool_forward_avx(benchmark::State& state) {
  tensor_t input_data, output_data;
  core::global_avepool_params params;
  std::tie(input_data, output_data, params) = get_bm_global_avepool_data();

  while (state.KeepRunning()) {
    kernels::global_avepool_op_avx(input_data, output_data, params, true);
  }
}
#endif

// Register the function as a benchmarks
BENCHMARK(bm_global_avepool_forward_internal)->Repetitions(1000);

#ifdef CNN_USE_AVX
BENCHMARK(bm_global_avepool_forward_avx)->Repetitions(1000);
#endif

}  // namespace benchmarks
}  // namespace tiny_dnn
