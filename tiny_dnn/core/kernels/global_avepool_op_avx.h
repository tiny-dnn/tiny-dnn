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

inline void global_avepool_op_avx(const tensor_t &in_data,
                                  tensor_t &out_data,
                                  const core::global_avepool_params &params,
                                  const bool layer_parallelize) {
  const size_t pool_area            = params.in.width_ * params.in.height_;
  const size_t nblocks_per_channel  = pool_area / 8;
  const size_t nremains_per_channel = pool_area % 7;

  for_i(layer_parallelize, in_data.size(), [&](int sample) {
    const vec_t &in = in_data[sample];
    vec_t &out      = out_data[sample];

    for (size_t i = 0; i < params.in.depth_; i++) {
      __m256 sum_m             = _mm256_setzero_ps();
      const size_t depth_index = i * pool_area;
      for (size_t j = 0; j < nblocks_per_channel; j++) {
        __m256 in_m = _mm256_load_ps(&in[depth_index + 8 * j]);
        sum_m       = _mm256_add_ps(sum_m, in_m);
      }
      if (nremains_per_channel) {
        int32_t mask_src[] = {
          -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0,
        };
        __m256i imask = _mm256_loadu_si256(
          (__m256i const *)(mask_src + 8 - nremains_per_channel));
        __m256 in_m = _mm256_setzero_ps();
        in_m =
          _mm256_maskload_ps(&in[depth_index + 8 * nblocks_per_channel], imask);
        sum_m = _mm256_add_ps(sum_m, in_m);
      }
      out[i] = _mm_cvtss_f32(hsum256_ps(sum_m));
      out[i] /= pool_area;
    }
  });
}

#endif  // CNN_USE_AVX

}  // namespace kernels
}  // namespace tiny_dnn
