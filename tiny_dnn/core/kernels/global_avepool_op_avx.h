/*
    Copyright (c) 2017, Taiga Nomi
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

namespace tiny_dnn {
namespace kernels {

#ifdef CNN_USE_AVX
#ifdef CNN_USE_DOUBLE

// double version
template <typename S1, typename S2>
inline void global_avepool_op_avx(const Tensor<double, S1> &in_data,
                                  Tensor<double, S2> &out_data,
                                  const core::global_avepool_params &params,
                                  const bool layer_parallelize) {
  const size_t pool_area            = params.in.width_ * params.in.height_;
  const size_t nblocks_per_channel  = pool_area / 4;
  const size_t nremains_per_channel = pool_area & 3;
  const size_t sample_num           = in_data.shape()[0];

  static const int64_t mask_src[] = {-1, -1, -1, -1, 0, 0, 0, 0};
  __m256i imask =
    _mm256_loadu_si256((__m256i const *)(mask_src + 4 - nremains_per_channel));
  __m128d one_m           = _mm_cvtsi64_sd(_mm_setzero_pd(), 1);
  __m128d pool_area_m     = _mm_cvtsi64_sd(_mm_setzero_pd(), pool_area);
  __m128d pool_area_inv_m = _mm_div_sd(one_m, pool_area_m);

  for_i(layer_parallelize, sample_num, [&](size_t sample) {
    const auto in = in_data.host_pointer(sample, 0);
    auto out      = out_data.host_pointer(sample, 0);
    for (size_t i = 0; i < params.in.depth_; i++) {
      const double *pin = &in[i * pool_area];
      __m256d sum0 =
        vectorize::accumulate<std::false_type>(pin, nblocks_per_channel);
      sum0 = _mm256_add_pd(
        sum0,
        _mm256_maskload_pd(pin + pool_area - nremains_per_channel, imask));
      _mm_store_sd(&out[i], _mm_mul_sd(hsum256_pd(sum0), pool_area_inv_m));
    }
  });
}

// double version
template <typename S1, typename S2>
inline void global_avepool_grad_op_avx(
  Tensor<double, S1> &prev_delta,
  const Tensor<double, S2> &curr_delta,
  const core::global_avepool_params &params,
  const bool layer_parallelize) {
  const size_t pool_area            = params.in.width_ * params.in.height_;
  const double pool_area_inv        = 1.0 / pool_area;
  const size_t nblocks_per_channel  = pool_area / 4;
  const size_t nremains_per_channel = pool_area & 3;
  const size_t sample_num           = prev_delta.shape()[0];

  int64_t mask_src[] = {-1, -1, -1, -1, 0, 0, 0, 0};
  __m256i imask =
    _mm256_loadu_si256((__m256i const *)(mask_src + 4 - nremains_per_channel));
  __m256d pool_area_inv_m = _mm256_set1_pd(pool_area_inv);

  for_i(layer_parallelize, sample_num, [&](size_t sample) {
    auto prev       = prev_delta.host_pointer(sample, 0);
    const auto curr = curr_delta.host_pointer(sample, 0);

    for (size_t i = 0; i < params.in.depth_; i++) {
      size_t j       = 0;
      __m256d prev_m = _mm256_broadcast_sd(&curr[i]);
      prev_m         = _mm256_mul_pd(prev_m, pool_area_inv_m);

      while (j < nblocks_per_channel) {
        _mm256_storeu_pd(&prev[i * pool_area + 4 * j], prev_m);
        ++j;
      }
      _mm256_maskstore_pd(&prev[i * pool_area + 4 * j], imask, prev_m);
    }
  });
}

#else  // CNN_USE_DOUBLE

// float version
template <typename S1, typename S2>
inline void global_avepool_op_avx(const Tensor<float, S1> &in_data,
                                  Tensor<float, S2> &out_data,
                                  const core::global_avepool_params &params,
                                  const bool layer_parallelize) {
  const size_t pool_area            = params.in.width_ * params.in.height_;
  const size_t nblocks_per_channel  = pool_area / 8;
  const size_t nremains_per_channel = pool_area & 7;
  const size_t sample_num           = in_data.shape()[0];

  static const int32_t mask_src[] = {
    -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  __m256i imask =
    _mm256_loadu_si256((__m256i const *)(mask_src + 8 - nremains_per_channel));
  __m128 one_m = _mm_cvtsi32_ss(_mm_setzero_ps(), 1);
  __m128 pool_area_m =
    _mm_cvtsi32_ss(_mm_setzero_ps(), static_cast<int>(pool_area));
  __m128 pool_area_inv_m = _mm_div_ss(one_m, pool_area_m);

  for_i(layer_parallelize, sample_num, [&](size_t sample) {
    const auto in = in_data.host_pointer(sample, 0);
    auto out      = out_data.host_pointer(sample, 0);

    for (size_t i = 0; i < params.in.depth_; i++) {
      const float *pin = &in[i * pool_area];
      __m256 sum0 =
        vectorize::accumulate<std::false_type>(pin, nblocks_per_channel);
      sum0 = _mm256_add_ps(
        sum0,
        _mm256_maskload_ps(pin + pool_area - nremains_per_channel, imask));
      _mm_store_ss(&out[i], _mm_mul_ss(hsum256_ps(sum0), pool_area_inv_m));
    }
  });
}

// float version
template <typename S1, typename S2>
inline void global_avepool_grad_op_avx(
  Tensor<float, S1> &prev_delta,
  const Tensor<float, S2> &curr_delta,
  const core::global_avepool_params &params,
  const bool layer_parallelize) {
  const size_t pool_area            = params.in.width_ * params.in.height_;
  const float pool_area_inv         = 1.0f / static_cast<float>(pool_area);
  const size_t nblocks_per_channel  = pool_area / 8;
  const size_t nremains_per_channel = pool_area & 7;
  const size_t sample_num           = prev_delta.shape()[0];

  int32_t mask_src[] = {
    -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  __m256i imask =
    _mm256_loadu_si256((__m256i const *)(mask_src + 8 - nremains_per_channel));

  for_i(layer_parallelize, sample_num, [&](size_t sample) {
    auto prev       = prev_delta.host_pointer(sample, 0);
    const auto curr = curr_delta.host_pointer(sample, 0);
    ;

    for (size_t i = 0; i < params.in.depth_; i++) {
      size_t j = 0;

      const __m256 prev_m = _mm256_set1_ps(curr[i] * pool_area_inv);

      while (j < nblocks_per_channel) {
        _mm256_storeu_ps(&prev[i * pool_area + 8 * j], prev_m);
        ++j;
      }
      _mm256_maskstore_ps(&prev[i * pool_area + 8 * j], imask, prev_m);
    }
  });
}

#endif  // CNN_USE_DOUBLE
#endif  // CNN_USE_AVX

}  // namespace kernels
}  // namespace tiny_dnn
