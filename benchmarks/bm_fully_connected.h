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

const size_t BM_CONST_IN_SIZE     = 64;
const size_t BM_CONST_OUT_SIZE    = 64;
const size_t BM_CONST_SAMPLE_SIZE = 1;

namespace tiny_dnn {
namespace benchmarks {

class BM_FullyConnectedOp : public bm::Fixture {
 public:
  void SetUp(const bm::State& state) {
    in_size  = state.range(0);
    out_size = state.range(1);
    samples  = state.range(2);

    in_data.reshape({1, in_size});
    out_data.reshape({1, out_size});
    weights.reshape({in_size * out_size});
    bias.reshape({out_size});

    uniform_rand(in_data.host_begin(), in_data.host_end(), -1, 1);
    uniform_rand(weights.host_begin(), weights.host_end(), -1, 1);
    uniform_rand(bias.host_begin(), bias.host_end(), -1, 1);
    uniform_rand(out_data.host_begin(), out_data.host_end(), -1, 1);
  }

  void TearDown(const bm::State&) {}

  size_t in_size, out_size, samples;
  Tensor<> in_data, out_data, weights, bias;
};

BENCHMARK_DEFINE_F(BM_FullyConnectedOp, internal)(bm::State& state) {
  while (state.KeepRunning()) {
    kernels::fully_connected_op_internal(in_data, weights, bias, out_data,
                                         false);
  }
}

BENCHMARK_REGISTER_F(FullyConnectedOp, internal)
  ->Args({16, 4, BM_CONST_SAMPLE_SIZE})
  ->Args({64, 4, BM_CONST_SAMPLE_SIZE})
  ->Args({256, 4, BM_CONST_SAMPLE_SIZE})
  ->Args({256, 16, BM_CONST_SAMPLE_SIZE})
  ->Args({256, 64, BM_CONST_SAMPLE_SIZE})
  ->Args({256, 256, BM_CONST_SAMPLE_SIZE});

BENCHMARK_DEFINE_F(BM_FullyConnectedOp, internal_parallel)(bm::State& state) {
  while (state.KeepRunning()) {
    kernels::fully_connected_op_internal(in_data, weights, bias, out_data,
                                         true);
  }
}

BENCHMARK_REGISTER_F(BM_FullyConnectedOp, internal_parallel)
  ->Args({BM_CONST_IN_SIZE, BM_CONST_OUT_SIZE, 1})
  ->Args({BM_CONST_IN_SIZE, BM_CONST_OUT_SIZE, 2})
  ->Args({BM_CONST_IN_SIZE, BM_CONST_OUT_SIZE, 4})
  ->Args({BM_CONST_IN_SIZE, BM_CONST_OUT_SIZE, 8})
  ->Args({BM_CONST_IN_SIZE, BM_CONST_OUT_SIZE, 16})
  ->Args({BM_CONST_IN_SIZE, BM_CONST_OUT_SIZE, 32});

#ifdef CNN_USE_AVX
BENCHMARK_DEFINE_F(BM_FullyConnectedOp, avx)(bm::State& state) {
  while (state.KeepRunning()) {
    kernels::fully_connected_op_avx(in_data, weights, bias, out_data, false);
  }
}

BENCHMARK_REGISTER_F(FullyConnectedOp, avx)
  ->Args({16, 4, BM_CONST_SAMPLE_SIZE})
  ->Args({64, 4, BM_CONST_SAMPLE_SIZE})
  ->Args({256, 4, BM_CONST_SAMPLE_SIZE})
  ->Args({256, 16, BM_CONST_SAMPLE_SIZE})
  ->Args({256, 64, BM_CONST_SAMPLE_SIZE})
  ->Args({256, 256, BM_CONST_SAMPLE_SIZE});

BENCHMARK_DEFINE_F(BM_FullyConnectedOp, avx_parallel)(bm::State& state) {
  while (state.KeepRunning()) {
    kernels::fully_connected_op_avx(in_data, weights, bias, out_data, true);
  }
}

BENCHMARK_REGISTER_F(BM_FullyConnectedOp, avx_parallel)
  ->Args({BM_CONST_IN_SIZE, BM_CONST_OUT_SIZE, 1})
  ->Args({BM_CONST_IN_SIZE, BM_CONST_OUT_SIZE, 2})
  ->Args({BM_CONST_IN_SIZE, BM_CONST_OUT_SIZE, 4})
  ->Args({BM_CONST_IN_SIZE, BM_CONST_OUT_SIZE, 8})
  ->Args({BM_CONST_IN_SIZE, BM_CONST_OUT_SIZE, 16})
  ->Args({BM_CONST_IN_SIZE, BM_CONST_OUT_SIZE, 32});

#endif

void bm_fully_connected_grad_internal(bm::State& state) {
  size_t in_size(state.range(0)), out_size(state.range(1));

  Tensor<> prev_out({1, in_size});
  Tensor<> weights({in_size * out_size});
  Tensor<> weights_grads({1, in_size * out_size}), bias_grads({1, out_size});
  Tensor<> curr_delta({1, out_size}), prev_delta({1, in_size});

  uniform_rand(prev_out.host_begin(), prev_out.host_end(), -1, 1);
  uniform_rand(weights.host_begin(), weights.host_end(), -1, 1);
  uniform_rand(weights_grads.host_begin(), weights_grads.host_end(), -1, 1);
  uniform_rand(bias_grads.host_begin(), bias_grads.host_end(), -1, 1);
  uniform_rand(curr_delta.host_begin(), curr_delta.host_end(), -1, 1);
  uniform_rand(prev_delta.host_begin(), prev_delta.host_end(), -1, 1);

  while (state.KeepRunning()) {
    kernels::fully_connected_op_internal(prev_out, weights, weights_grads,
                                         bias_grads, curr_delta, prev_delta,
                                         false);
  }
}

void bm_fully_connected_grad_internal_parallelized(bm::State& state) {
  size_t in_size(64), out_size(64), samples(state.range(0));

  Tensor<> prev_out({samples, in_size});
  Tensor<> weights({in_size * out_size});
  Tensor<> weights_grads({samples, in_size * out_size}),
    bias_grads({samples, out_size});
  Tensor<> curr_delta({samples, out_size}), prev_delta({samples, in_size});

  uniform_rand(prev_out.host_begin(), prev_out.host_end(), -1, 1);
  uniform_rand(weights.host_begin(), weights.host_end(), -1, 1);
  uniform_rand(weights_grads.host_begin(), weights_grads.host_end(), -1, 1);
  uniform_rand(bias_grads.host_begin(), bias_grads.host_end(), -1, 1);
  uniform_rand(curr_delta.host_begin(), curr_delta.host_end(), -1, 1);
  uniform_rand(prev_delta.host_begin(), prev_delta.host_end(), -1, 1);

  while (state.KeepRunning()) {
    kernels::fully_connected_op_internal(prev_out, weights, weights_grads,
                                         bias_grads, curr_delta, prev_delta,
                                         true);
  }
}

BENCHMARK(bm_fully_connected_grad_internal)
  ->Args({16, 4})
  ->Args({64, 4})
  ->Args({256, 4})
  ->Args({256, 16})
  ->Args({256, 64})
  ->Args({256, 256});

BENCHMARK(bm_fully_connected_grad_internal_parallelized)
  ->RangeMultiplier(2)
  ->Range(1, 32);

#ifdef CNN_USE_AVX
void bm_fully_connected_grad_avx(bm::State& state) {
  size_t in_size(state.range(0)), out_size(state.range(1));

  Tensor<> prev_out({1, in_size});
  Tensor<> weights({in_size * out_size});
  Tensor<> weights_grads({1, in_size * out_size}), bias_grads({1, out_size});
  Tensor<> curr_delta({1, out_size}), prev_delta({1, in_size});

  uniform_rand(prev_out.host_begin(), prev_out.host_end(), -1, 1);
  uniform_rand(weights.host_begin(), weights.host_end(), -1, 1);
  uniform_rand(weights_grads.host_begin(), weights_grads.host_end(), -1, 1);
  uniform_rand(bias_grads.host_begin(), bias_grads.host_end(), -1, 1);
  uniform_rand(curr_delta.host_begin(), curr_delta.host_end(), -1, 1);
  uniform_rand(prev_delta.host_begin(), prev_delta.host_end(), -1, 1);

  while (state.KeepRunning()) {
    kernels::fully_connected_op_avx(prev_out, weights, weights_grads,
                                    bias_grads, curr_delta, prev_delta, false);
  }
}

void bm_fully_connected_grad_avx_parallelized(bm::State& state) {
  size_t in_size(64), out_size(64), samples(state.range(0));

  Tensor<> prev_out({samples, in_size});
  Tensor<> weights({in_size * out_size});
  Tensor<> weights_grads({samples, in_size * out_size}),
    bias_grads({samples, out_size});
  Tensor<> curr_delta({samples, out_size}), prev_delta({samples, in_size});

  uniform_rand(prev_out.host_begin(), prev_out.host_end(), -1, 1);
  uniform_rand(weights.host_begin(), weights.host_end(), -1, 1);
  uniform_rand(weights_grads.host_begin(), weights_grads.host_end(), -1, 1);
  uniform_rand(bias_grads.host_begin(), bias_grads.host_end(), -1, 1);
  uniform_rand(curr_delta.host_begin(), curr_delta.host_end(), -1, 1);
  uniform_rand(prev_delta.host_begin(), prev_delta.host_end(), -1, 1);

  while (state.KeepRunning()) {
    kernels::fully_connected_op_avx(prev_out, weights, weights_grads,
                                    bias_grads, curr_delta, prev_delta, true);
  }
}

BENCHMARK(bm_fully_connected_grad_avx)
  ->Args({16, 4})
  ->Args({64, 4})
  ->Args({256, 4})
  ->Args({256, 16})
  ->Args({256, 64})
  ->Args({256, 256});

BENCHMARK(bm_fully_connected_grad_avx_parallelized)
  ->RangeMultiplier(2)
  ->Range(1, 32);
#endif

}  // namespace benchmarks
}  // namespace tiny_dnn
