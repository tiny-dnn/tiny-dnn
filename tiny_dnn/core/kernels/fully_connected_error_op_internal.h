/*
    Copyright (c) 2016, Taiga Nomi, Edgar Riba
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once
#define ERROR_RATE 0.001
#include "tiny_dnn/core/params/fully_params.h"
#include <assert.h>
#include <random>
#include <bitset>
namespace tiny_dnn {
namespace kernels {

template<size_t size>
typename std::bitset<size> random_bitset( double p = 0.5) {
	typename std::bitset<size> bits;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::bernoulli_distribution d(p);
	for ( int n = 0; n < size; n++ ) {
		bits[n] = d(gen);
		//bits[n]= 0;
	}
	return bits;
}

double genErrorDouble(double errorRate, double orig) {
	union Flip
	{
		double input;
		long long output;
	} data, ret;
	data.input = orig;

	auto mask = random_bitset<sizeof(double)>(errorRate);
	std::bitset<sizeof(double)> bits(data.output);
	ret.output = ( bits ^ mask ).to_ullong();
	return ret.input;
}

inline void fully_connected_error_op_internal(const tensor_t &in_data,
                                        const vec_t &W,
                                        const vec_t &bias,
                                        tensor_t &out_data,
                                        const fully_params &params,
                                        const bool layer_parallelize) {
  for_i(layer_parallelize, in_data.size(), [&](int sample) {
    //assert(false);
    const vec_t &in = in_data[sample];
    vec_t &out      = out_data[sample];

    for (serial_size_t i = 0; i < params.out_size_; i++) {
      out[i] = float_t{0};
      for (serial_size_t c = 0; c < params.in_size_; c++) {
        out[i] += W[c * params.out_size_ + i] * in[c];
	double error = genErrorDouble(ERROR_RATE, out[i]);
	out[i] = error;
      }

      if (params.has_bias_) {
        out[i] += bias[i];
	double error = genErrorDouble(ERROR_RATE, out[i]);
	out[i] = error;
      }
    }
  });
}

inline void fully_connected_error_op_internal(const tensor_t &prev_out,
                                        const vec_t &W,
                                        tensor_t &dW,
                                        tensor_t &db,
                                        tensor_t &curr_delta,
                                        tensor_t &prev_delta,
                                        const fully_params &params,
                                        const bool layer_parallelize) {
  assert(false);
  for (serial_size_t sample = 0; sample < prev_out.size(); sample++) {
    for (serial_size_t c = 0; c < params.in_size_; c++) {
      // propagate delta to previous layer
      // prev_delta[c] += current_delta[r] * W_[c * out_size_ + r]
      prev_delta[sample][c] += vectorize::dot(
        &curr_delta[sample][0], &W[c * params.out_size_], params.out_size_);
    }

    for_(layer_parallelize, 0, size_t(params.out_size_),
         [&](const blocked_range &r) {
           // accumulate weight-step using delta
           // dW[c * out_size + i] += current_delta[i] * prev_out[c]
           for (serial_size_t c = 0; c < params.in_size_; c++) {
             vectorize::muladd(&curr_delta[sample][r.begin()],
                               prev_out[sample][c], r.end() - r.begin(),
                               &dW[sample][c * params.out_size_ + r.begin()]);
           }

           if (params.has_bias_) {
             // vec_t& db = *in_grad[2];
             for (int i = r.begin(); i < r.end(); i++) {
               db[sample][i] += curr_delta[sample][i];
             }
           }
         });
  }
}

}  // namespace kernels
}  // namespace tiny_dnn
