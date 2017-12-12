/*
    Copyright (c) 2017, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/util/product.h"

TEST(vectorize, accumulate) {
  size_t sz = 64;
  std::vector<float> vec(sz);
  std::fill(vec.begin(),vec.end(), 1);

  __m128 one_m = _mm_cvtsi32_ss(_mm_setzero_ps(), 1);
  __m128 sz_m = _mm_cvtsi32_ss(_mm_setzero_ps(), sz);
  __m128 sz_inv_m = _mm_div_ss(one_m, sz_m);
  __m256 sum0 = vectorize::accumulate<std::false_type>(&vec[0], sz/8);
  float_t out = _mm_cvtss_f32(_mm_mul_ss(hsum256_ps(sum0), sz_inv_m));
  ASSERT_NEAR(out, 1, std::numeric_limits<float>::epsilon());
}