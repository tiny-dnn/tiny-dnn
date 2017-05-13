/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/kernels/fully_connected_op_internal.h"

namespace tiny_dnn {
namespace kernels {

#ifdef CNN_USE_AVX

template <class E1,
          class E2,
          class E3,
          class E4,
          value_is_float<E1> * = nullptr,
          are_all_xexpr<E1, E2, E3, E4> * = nullptr>
inline void avx_fully_connected_forward_kernel(const E1 &in_data,
                                               const E2 &W,
                                               const E3 &bias,
                                               E4 &out_data,
                                               const fully_params &params,
                                               const bool layer_parallelize) {
  auto in_shape    = in_data.shape();
  auto num_samples = in_shape[0];

  if (params.has_bias_) {
    size_t nblocks  = params.out_size_ / 8;
    size_t nremains = params.out_size_ & 7;
    if (nremains) {
      int32_t mask_src[] = {
        -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0,
      };
      __m256i imask =
        _mm256_loadu_si256((__m256i const *)(mask_src + 8 - nremains));
      for_i(layer_parallelize, num_samples, [&](int sample) {
        const auto in = xt::view(in_data, sample, xt::all());
        auto out      = xt::view(out_data, sample, xt::all());
        {
          for (size_t i = 0; i < nblocks; ++i) {
            __m256 b = _mm256_loadu_ps(&bias[8 * i]);
            _mm256_storeu_ps(&out[8 * i], b);
          }
          auto b = _mm256_maskload_ps(&bias[8 * nblocks], imask);
          _mm256_maskstore_ps(&out[8 * nblocks], imask, b);
        }
        for (serial_size_t c = 0; c < params.in_size_; c++) {
          auto in_val     = _mm256_set1_ps(in[c]);
          const float *pW = &W[c * params.out_size_];
          for (size_t i = 0; i < nblocks / 2; ++i) {
            __m256 sum0 = _mm256_loadu_ps(&out[16 * i]);
            __m256 sum1 = _mm256_loadu_ps(&out[16 * i + 8]);
            __m256 w0   = _mm256_loadu_ps(pW + 16 * i);
            __m256 w1   = _mm256_loadu_ps(pW + 16 * i + 8);
            sum0        = madd256_ps(w0, in_val, sum0);
            sum1        = madd256_ps(w1, in_val, sum1);
            _mm256_storeu_ps(&out[16 * i], sum0);
            _mm256_storeu_ps(&out[16 * i + 8], sum1);
          }
          if (nblocks & 1) {
            __m256 sum0 = _mm256_loadu_ps(&out[nblocks / 2 * 16]);
            __m256 w0   = _mm256_loadu_ps(pW + nblocks / 2 * 16);
            sum0        = madd256_ps(w0, in_val, sum0);
            _mm256_storeu_ps(&out[nblocks / 2 * 16], sum0);
          }
          __m256 sum = _mm256_maskload_ps(&out[8 * nblocks], imask);
          __m256 w   = _mm256_maskload_ps(pW + 8 * nblocks, imask);
          sum        = madd256_ps(w, in_val, sum);
          _mm256_maskstore_ps(&out[8 * nblocks], imask, sum);
        }
      });
    } else {
      for_i(layer_parallelize, num_samples, [&](int sample) {
        const auto in = xt::view(in_data, sample, xt::all());
        auto out      = xt::view(out_data, sample, xt::all());
        for (size_t i = 0; i < nblocks; ++i) {
          __m256 b = _mm256_loadu_ps(&bias[8 * i]);
          _mm256_storeu_ps(&out[8 * i], b);
        }
        for (serial_size_t c = 0; c < params.in_size_; c++) {
          auto in_val     = _mm256_set1_ps(in[c]);
          const float *pW = &W[c * params.out_size_];
          for (size_t i = 0; i < nblocks; ++i) {
            __m256 sum = _mm256_loadu_ps(&out[8 * i]);
            __m256 w   = _mm256_loadu_ps(pW + 8 * i);
            sum        = madd256_ps(w, in_val, sum);
            _mm256_storeu_ps(&out[8 * i], sum);
          }
        }
      });
    }
  } else {
    for_i(layer_parallelize, num_samples, [&](int sample) {
      const auto in = xt::view(in_data, sample, xt::all());
      auto out      = xt::view(out_data, sample, xt::all());
      for (serial_size_t i = 0; i < params.out_size_; i++) {
        float sum = 0.0f;
        for (serial_size_t c = 0; c < params.in_size_; c++) {
          sum += W[c * params.out_size_ + i] * in[c];
        }
        out[i] = sum;
      }
    });
  }
}

template <class E1,
          class E2,
          class E3,
          class E4,
          value_is_double<E1> * = nullptr,
          are_all_xexpr<E1, E2, E3, E4> * = nullptr>
inline void avx_fully_connected_forward_kernel(const E1 &in_data,
                                               const E2 &W,
                                               const E3 &bias,
                                               E4 &out_data,
                                               const fully_params &params,
                                               const bool layer_parallelize) {
  // fallback to tiny-backend when float_t is double
  fully_connected_op_internal(in_data, W, bias, out_data, params,
                              layer_parallelize);
}

template <class E1,
          class E2,
          class E3,
          class E4,
          class E5,
          class E6,
          value_is_float<E1> * = nullptr,
          are_all_xexpr<E1, E2, E3, E4, E5, E6> * = nullptr>
inline void avx_fully_connected_back_kernel(const E1 &prev_out,
                                            const E2 &W,
                                            E3 &dW,
                                            E4 &db,
                                            E5 &curr_delta,
                                            E6 &prev_delta,
                                            const fully_params &params,
                                            const bool layer_parallelize) {
  auto prev_out_shape = prev_out.shape();
  auto num_samples    = prev_out_shape[0];
  if (params.has_bias_) {
    for (serial_size_t sample = 0; sample < num_samples; sample++) {
      auto prev_delta2     = xt::view(prev_delta, sample, xt::all());
      auto curr_delta2     = xt::view(curr_delta, sample, xt::all());
      const auto prev_out2 = xt::view(prev_out, sample, xt::all());
      auto dW2             = xt::view(dW, sample, xt::all());
      auto db2             = xt::view(db, sample, xt::all());
      for (serial_size_t c = 0; c < params.in_size_; c++) {
        // propagate delta to previous layer
        // prev_delta[c] += current_delta[r] * W_[c * out_size_ + r]
        prev_delta2[c] += vectorize::dot(
          &curr_delta2[0], &W[c * params.out_size_], params.out_size_);
      }
      for_(layer_parallelize, 0, size_t(params.out_size_),
           [&](const blocked_range &r) {
             // accumulate weight-step using delta
             // dW[c * out_size + i] += current_delta[i] * prev_out[c]
             size_t len = r.end() - r.begin();
             for (serial_size_t c = 0; c < params.in_size_; c++) {
               vectorize::muladd(&curr_delta2[r.begin()], prev_out2[c], len,
                                 &dW2[c * params.out_size_ + r.begin()]);
             }
             // vec_t& db = *in_grad[2];
             vectorize::reduce(&curr_delta2[r.begin()], len, &db2[r.begin()]);
           });
    }
  } else {
    for (serial_size_t sample = 0; sample < num_samples; sample++) {
      auto prev_delta2     = xt::view(prev_delta, sample, xt::all());
      auto curr_delta2     = xt::view(curr_delta, sample, xt::all());
      const auto prev_out2 = xt::view(prev_out, sample, xt::all());
      auto dW2             = xt::view(dW, sample, xt::all());
      for (serial_size_t c = 0; c < params.in_size_; c++) {
        // propagate delta to previous layer
        // prev_delta[c] += current_delta[r] * W_[c * out_size_ + r]
        prev_delta2[c] += vectorize::dot(
          &curr_delta2[0], &W[c * params.out_size_], params.out_size_);
      }
      for_(layer_parallelize, 0, size_t(params.out_size_),
           [&](const blocked_range &r) {
             // accumulate weight-step using delta
             // dW[c * out_size + i] += current_delta[i] * prev_out[c]
             size_t len = r.end() - r.begin();
             for (serial_size_t c = 0; c < params.in_size_; c++) {
               vectorize::muladd(&curr_delta2[r.begin()], prev_out2[c], len,
                                 &dW2[c * params.out_size_ + r.begin()]);
             }
           });
    }
  }
}

template <class E1,
          class E2,
          class E3,
          class E4,
          class E5,
          class E6,
          value_is_double<E1> * = nullptr,
          are_all_xexpr<E1, E2, E3, E4, E5, E6> * = nullptr>
inline void avx_fully_connected_back_kernel(const E1 &prev_out,
                                            const E2 &W,
                                            E3 &dW,
                                            E4 &db,
                                            E5 &curr_delta,
                                            E6 &prev_delta,
                                            const fully_params &params,
                                            const bool layer_parallelize) {
  // fallback to tiny-backend when float_t is double
  fully_connected_op_internal(prev_out, W, dW, db, curr_delta, prev_delta,
                              params, layer_parallelize);
}

#endif  // CNN_USE_AVX
template <class E1,
          class E2,
          class E3,
          class E4,
          are_all_xexpr<E1, E2, E3, E4> * = nullptr>
inline void fully_connected_op_avx(E1 &in_data,
                                   E2 W,
                                   E3 bias, //TODO
                                   E4 &out_data,
                                   const fully_params &params,
                                   const bool layer_parallelize) {
#ifdef CNN_USE_AVX
  avx_fully_connected_forward_kernel(in_data, W, bias, out_data, params,
                                     layer_parallelize);
#else
  CNN_UNREFERENCED_PARAMETER(in_data);
  CNN_UNREFERENCED_PARAMETER(W);
  CNN_UNREFERENCED_PARAMETER(bias);
  CNN_UNREFERENCED_PARAMETER(out_data);
  CNN_UNREFERENCED_PARAMETER(params);
  CNN_UNREFERENCED_PARAMETER(layer_parallelize);
  throw nn_error("TinyDNN has not been compiled with AVX support.");
#endif
}

template <class E1,
          class E2,
          class E3,
          class E4,
          class E5,
          class E6,
          are_all_xexpr<E1, E2, E3, E4, E5, E6> * = nullptr>
inline void fully_connected_op_avx(E1 &prev_out,
                                   E2 W,
                                   E3 &dW,
                                   E4 &db,
                                   E5 &curr_delta,
                                   E6 &prev_delta,
                                   const fully_params &params,
                                   const bool layer_parallelize) {
#ifdef CNN_USE_AVX
  avx_fully_connected_back_kernel(prev_out, W, dW, db, curr_delta, prev_delta,
                                  params, layer_parallelize);
#else
  CNN_UNREFERENCED_PARAMETER(prev_out);
  CNN_UNREFERENCED_PARAMETER(W);
  CNN_UNREFERENCED_PARAMETER(dW);
  CNN_UNREFERENCED_PARAMETER(db);
  CNN_UNREFERENCED_PARAMETER(curr_delta);
  CNN_UNREFERENCED_PARAMETER(prev_delta);
  CNN_UNREFERENCED_PARAMETER(params);
  CNN_UNREFERENCED_PARAMETER(layer_parallelize);
  throw nn_error("TinyDNN has not been compiled with AVX support.");
#endif
}

}  // namespace kernels
}  // namespace tiny_dnn
