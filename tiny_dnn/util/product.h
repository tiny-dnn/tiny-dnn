/*
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#if defined(CNN_USE_SSE) || defined(CNN_USE_AVX)
#include <immintrin.h>
#endif
#include <cassert>
#include <cstdint>
#include <numeric>

#ifdef CNN_USE_AVX
#include "tiny_dnn/core/kernels/avx_kernel_common.h"
#endif

namespace vectorize {
namespace detail {

// traits
template <typename T>
struct scalar_generic {
  typedef T register_type;
  typedef T value_type;
  enum { unroll_size = 1 };
  static register_type set1(const value_type &x) { return x; }
  static register_type zero() { return register_type(0); }
  static register_type mul(const register_type &v1, const register_type &v2) {
    return v1 * v2;
  }
  static register_type add(const register_type &v1, const register_type &v2) {
    return v1 + v2;
  }
  static register_type madd(const register_type &v1,
                            const register_type &v2,
                            const register_type &v3) {
    return v1 * v2 + v3;
  }

  template <typename aligned>
  static register_type load(const value_type *px) {
    return *px;
  }
  template <typename aligned>
  static void store(value_type *px, const register_type &v) {
    *px = v;
  }

  static value_type resemble(const register_type &x) { return x; }

  static bool is_aligned(value_type *p) { return true; }
};

#ifdef CNN_USE_SSE

struct float_sse {
  typedef __m128 register_type;
  typedef float value_type;
  enum { unroll_size = 4 };
  static register_type set1(const value_type &x) { return _mm_set1_ps(x); }
  static register_type zero() { return _mm_setzero_ps(); }
  static register_type mul(const register_type &v1, const register_type &v2) {
    return _mm_mul_ps(v1, v2);
  }
  static register_type add(const register_type &v1, const register_type &v2) {
    return _mm_add_ps(v1, v2);
  }
  static register_type madd(const register_type &v1,
                            const register_type &v2,
                            const register_type &v3) {
    return _mm_add_ps(_mm_mul_ps(v1, v2), v3);
  }

  template <typename aligned>
  static register_type load(const value_type *px);

  template <typename aligned>
  static void store(value_type *px, const register_type &v);

  static value_type resemble(const register_type &x) {
    alignas(16) float tmp[4];
    _mm_store_ps(tmp, x);
    return tmp[0] + tmp[1] + tmp[2] + tmp[3];
  }
  static bool is_aligned(value_type *p) {
    return reinterpret_cast<uintptr_t>(p) % 16 == 0;
  }
};

template <>
inline __m128 float_sse::load<std::true_type>(const float *px) {
  return _mm_load_ps(px);
}
template <>
inline __m128 float_sse::load<std::false_type>(const float *px) {
  return _mm_loadu_ps(px);
}

template <>
inline void float_sse::store<std::true_type>(float *px, const __m128 &v) {
  _mm_store_ps(px, v);
}
template <>
inline void float_sse::store<std::false_type>(float *px, const __m128 &v) {
  _mm_storeu_ps(px, v);
}

struct double_sse {
  typedef __m128d register_type;
  typedef double value_type;
  enum { unroll_size = 2 };
  static register_type set1(const value_type &x) { return _mm_set1_pd(x); }
  static register_type zero() { return _mm_setzero_pd(); }
  static register_type mul(const register_type &v1, const register_type &v2) {
    return _mm_mul_pd(v1, v2);
  }
  static register_type add(const register_type &v1, const register_type &v2) {
    return _mm_add_pd(v1, v2);
  }
  static register_type madd(const register_type &v1,
                            const register_type &v2,
                            const register_type &v3) {
    return _mm_add_pd(_mm_mul_pd(v1, v2), v3);
  }

  template <typename aligned>
  static register_type load(const value_type *px);

  template <typename aligned>
  static void store(value_type *px, const register_type &v);

  static value_type resemble(const register_type &x) {
    alignas(16) double tmp[2];
    _mm_store_pd(tmp, x);
    return tmp[0] + tmp[1];
  }

  static bool is_aligned(value_type *p) {
    return reinterpret_cast<uintptr_t>(p) % 16 == 0;
  }
};

template <>
inline __m128d double_sse::load<std::true_type>(const double *px) {
  return _mm_load_pd(px);
}
template <>
inline __m128d double_sse::load<std::false_type>(const double *px) {
  return _mm_loadu_pd(px);
}

template <>
inline void double_sse::store<std::true_type>(double *px, const __m128d &v) {
  _mm_store_pd(px, v);
}
template <>
inline void double_sse::store<std::false_type>(double *px, const __m128d &v) {
  _mm_storeu_pd(px, v);
}

#endif  // CNN_USE_SSE

#ifdef CNN_USE_AVX

struct float_avx {
  typedef __m256 register_type;
  typedef float value_type;
  enum { unroll_size = 8 };
  static register_type set1(const value_type &x) { return _mm256_set1_ps(x); }
  static register_type zero() { return _mm256_setzero_ps(); }
  static register_type mul(const register_type &v1, const register_type &v2) {
    return _mm256_mul_ps(v1, v2);
  }
  static register_type add(const register_type &v1, const register_type &v2) {
    return _mm256_add_ps(v1, v2);
  }
#ifdef CNN_USE_AVX2
  static register_type madd(const register_type &v1,
                            const register_type &v2,
                            const register_type &v3) {
    return _mm256_fmadd_ps(v1, v2, v3);
  }
#else
  static register_type madd(const register_type &v1,
                            const register_type &v2,
                            const register_type &v3) {
    return _mm256_add_ps(_mm256_mul_ps(v1, v2), v3);
  }
#endif

  template <typename aligned>
  static register_type load(const value_type *px);

  template <typename aligned>
  static void store(value_type *px, const register_type &v);

  static value_type resemble(const register_type &x) {
    return _mm_cvtss_f32(hsum256_ps(x));
  }
  static bool is_aligned(value_type *p) {
    return reinterpret_cast<uintptr_t>(p) % 32 == 0;
  }
};

template <>
inline __m256 float_avx::load<std::true_type>(const float *px) {
  return _mm256_load_ps(px);
}
template <>
inline __m256 float_avx::load<std::false_type>(const float *px) {
  return _mm256_loadu_ps(px);
}

template <>
inline void float_avx::store<std::true_type>(float *px, const __m256 &v) {
  _mm256_store_ps(px, v);
}
template <>
inline void float_avx::store<std::false_type>(float *px, const __m256 &v) {
  _mm256_storeu_ps(px, v);
}

struct double_avx {
  typedef __m256d register_type;
  typedef double value_type;
  enum { unroll_size = 4 };
  static register_type set1(const value_type &x) { return _mm256_set1_pd(x); }
  static register_type zero() { return _mm256_setzero_pd(); }
  static register_type mul(const register_type &v1, const register_type &v2) {
    return _mm256_mul_pd(v1, v2);
  }
  static register_type add(const register_type &v1, const register_type &v2) {
    return _mm256_add_pd(v1, v2);
  }
#ifdef CNN_USE_AVX2
  static register_type madd(const register_type &v1,
                            const register_type &v2,
                            const register_type &v3) {
    return _mm256_fmadd_pd(v1, v2, v3);
  }
#else
  static register_type madd(const register_type &v1,
                            const register_type &v2,
                            const register_type &v3) {
    return _mm256_add_pd(_mm256_mul_pd(v1, v2), v3);
  }
#endif

  template <typename aligned>
  static register_type load(const value_type *px);

  template <typename aligned>
  static void store(value_type *px, const register_type &v);

  static value_type resemble(const register_type &x) {
    alignas(32) double tmp[4];
    _mm256_store_pd(tmp, x);
    return std::accumulate(tmp, tmp + 4, 0.0);
  }
  static bool is_aligned(value_type *p) {
    return reinterpret_cast<uintptr_t>(p) % 32 == 0;
  }
};

template <>
inline __m256d double_avx::load<std::true_type>(const double *px) {
  return _mm256_load_pd(px);
}
template <>
inline __m256d double_avx::load<std::false_type>(const double *px) {
  return _mm256_loadu_pd(px);
}

template <>
inline void double_avx::store<std::true_type>(double *px, const __m256d &v) {
  _mm256_store_pd(px, v);
}
template <>
inline void double_avx::store<std::false_type>(double *px, const __m256d &v) {
  _mm256_storeu_pd(px, v);
}

#endif  // CNN_USE_AVX

// generic dot-product
template <typename T, typename f1_aligned, typename f2_aligned>
inline typename T::value_type dot_product(const typename T::value_type *f1,
                                          const typename T::value_type *f2,
                                          std::size_t size) {
  typename T::register_type r0 = T::zero();
  typename T::register_type r1 = T::zero();
  typename T::register_type r2 = T::zero();
  typename T::register_type r3 = T::zero();
  auto sz                      = T::unroll_size;
  auto sz4                     = T::unroll_size * 4;
  auto n4                      = size / sz4;
  auto n1                      = (size % sz4) / sz;
  auto remain                  = size % sz;
  for (size_t i = 0; i < n4; ++i) {
    auto s10 = T::template load<f1_aligned>(&f1[i * sz4 + sz * 0]);
    auto s11 = T::template load<f1_aligned>(&f1[i * sz4 + sz * 1]);
    auto s12 = T::template load<f1_aligned>(&f1[i * sz4 + sz * 2]);
    auto s13 = T::template load<f1_aligned>(&f1[i * sz4 + sz * 3]);
    auto s20 = T::template load<f2_aligned>(&f2[i * sz4 + sz * 0]);
    auto s21 = T::template load<f2_aligned>(&f2[i * sz4 + sz * 1]);
    auto s22 = T::template load<f2_aligned>(&f2[i * sz4 + sz * 2]);
    auto s23 = T::template load<f2_aligned>(&f2[i * sz4 + sz * 3]);
    r0       = T::madd(s10, s20, r0);
    r1       = T::madd(s11, s21, r1);
    r2       = T::madd(s12, s22, r2);
    r3       = T::madd(s13, s23, r3);
  }
  size_t idx = n4 * sz4;
  for (size_t i = 0; i < n1; ++i) {
    auto s1 = T::template load<f1_aligned>(&f1[idx + i * sz]);
    auto s2 = T::template load<f2_aligned>(&f2[idx + i * sz]);
    r0      = T::madd(s1, s2, r0);
  }
  r0                         = T::add(r0, r1);
  r2                         = T::add(r2, r3);
  r0                         = T::add(r0, r2);
  typename T::value_type sum = T::resemble(r0);
  idx += n1 * sz;
  for (size_t i = 0; i < remain; ++i) {
    sum += f1[idx + i] * f2[idx + i];
  }
  return sum;
}

template <typename T, typename dst_aligned>
inline void add(typename T::value_type c,
                std::size_t size,
                typename T::value_type *dst) {
  typename T::register_type c2 = T::set1(c);
  auto sz                      = T::unroll_size;
  auto sz4                     = T::unroll_size * 4;
  auto n4                      = size / sz4;
  auto n1                      = (size % sz4) / sz;
  auto remain                  = size % sz;
  for (size_t i = 0; i < n4; ++i) {
    auto d0 = T::template load<dst_aligned>(&dst[i * sz4 + sz * 0]);
    auto d1 = T::template load<dst_aligned>(&dst[i * sz4 + sz * 1]);
    auto d2 = T::template load<dst_aligned>(&dst[i * sz4 + sz * 2]);
    auto d3 = T::template load<dst_aligned>(&dst[i * sz4 + sz * 3]);
    d0      = T::add(c2, d0);
    d1      = T::add(c2, d1);
    d2      = T::add(c2, d2);
    d3      = T::add(c2, d3);
    T::template store<dst_aligned>(&dst[i * sz4 + sz * 0], d0);
    T::template store<dst_aligned>(&dst[i * sz4 + sz * 1], d1);
    T::template store<dst_aligned>(&dst[i * sz4 + sz * 2], d2);
    T::template store<dst_aligned>(&dst[i * sz4 + sz * 3], d3);
  }
  size_t idx = n4 * sz4;
  for (size_t i = 0; i < n1; ++i) {
    auto d = T::template load<dst_aligned>(&dst[idx + i * sz]);
    d      = T::add(c2, d);
    T::template store<dst_aligned>(&dst[idx + i * sz], d);
  }
  idx += n1 * sz;
  for (size_t i = 0; i < remain; ++i) {
    dst[idx + i] += c;
  }
}

template <typename T, typename src_aligned, typename dst_aligned>
inline void add(const typename T::value_type *src,
                std::size_t size,
                typename T::value_type *dst) {
  auto sz     = T::unroll_size;
  auto sz4    = T::unroll_size * 4;
  auto n4     = size / sz4;
  auto n1     = (size % sz4) / sz;
  auto remain = size % sz;
  for (size_t i = 0; i < n4; ++i) {
    auto d0 = T::template load<dst_aligned>(&dst[i * sz4 + sz * 0]);
    auto d1 = T::template load<dst_aligned>(&dst[i * sz4 + sz * 1]);
    auto d2 = T::template load<dst_aligned>(&dst[i * sz4 + sz * 2]);
    auto d3 = T::template load<dst_aligned>(&dst[i * sz4 + sz * 3]);
    auto s0 = T::template load<src_aligned>(&src[i * sz4 + sz * 0]);
    auto s1 = T::template load<src_aligned>(&src[i * sz4 + sz * 1]);
    auto s2 = T::template load<src_aligned>(&src[i * sz4 + sz * 2]);
    auto s3 = T::template load<src_aligned>(&src[i * sz4 + sz * 3]);
    d0      = T::add(s0, d0);
    d1      = T::add(s1, d1);
    d2      = T::add(s2, d2);
    d3      = T::add(s3, d3);
    T::template store<dst_aligned>(&dst[i * sz4 + sz * 0], d0);
    T::template store<dst_aligned>(&dst[i * sz4 + sz * 1], d1);
    T::template store<dst_aligned>(&dst[i * sz4 + sz * 2], d2);
    T::template store<dst_aligned>(&dst[i * sz4 + sz * 3], d3);
  }
  size_t idx = n4 * sz4;
  for (size_t i = 0; i < n1; ++i) {
    auto d = T::template load<dst_aligned>(&dst[idx + i * sz]);
    auto s = T::template load<src_aligned>(&src[idx + i * sz]);
    d      = T::add(s, d);
    T::template store<dst_aligned>(&dst[idx + i * sz], d);
  }
  idx += n1 * sz;
  for (size_t i = 0; i < remain; ++i) {
    dst[idx + i] += src[idx + i];
  }
}

template <typename T, typename src_aligned, typename dst_aligned>
inline void muladd(const typename T::value_type *src,
                   typename T::value_type c,
                   std::size_t size,
                   typename T::value_type *dst) {
  auto factor = T::set1(c);
  auto sz     = T::unroll_size;
  auto sz4    = T::unroll_size * 4;
  auto n4     = size / sz4;
  auto n1     = (size % sz4) / sz;
  auto remain = size % sz;
  for (size_t i = 0; i < n4; ++i) {
    auto d0 = T::template load<dst_aligned>(&dst[i * sz4 + sz * 0]);
    auto d1 = T::template load<dst_aligned>(&dst[i * sz4 + sz * 1]);
    auto d2 = T::template load<dst_aligned>(&dst[i * sz4 + sz * 2]);
    auto d3 = T::template load<dst_aligned>(&dst[i * sz4 + sz * 3]);
    auto s0 = T::template load<src_aligned>(&src[i * sz4 + sz * 0]);
    auto s1 = T::template load<src_aligned>(&src[i * sz4 + sz * 1]);
    auto s2 = T::template load<src_aligned>(&src[i * sz4 + sz * 2]);
    auto s3 = T::template load<src_aligned>(&src[i * sz4 + sz * 3]);
    d0      = T::madd(s0, factor, d0);
    d1      = T::madd(s1, factor, d1);
    d2      = T::madd(s2, factor, d2);
    d3      = T::madd(s3, factor, d3);
    T::template store<dst_aligned>(&dst[i * sz4 + sz * 0], d0);
    T::template store<dst_aligned>(&dst[i * sz4 + sz * 1], d1);
    T::template store<dst_aligned>(&dst[i * sz4 + sz * 2], d2);
    T::template store<dst_aligned>(&dst[i * sz4 + sz * 3], d3);
  }
  size_t idx = n4 * sz4;
  for (size_t i = 0; i < n1; ++i) {
    auto d = T::template load<dst_aligned>(&dst[idx + i * sz]);
    auto s = T::template load<src_aligned>(&src[idx + i * sz]);
    d      = T::madd(s, factor, d);
    T::template store<dst_aligned>(&dst[idx + i * sz], d);
  }
  idx += n1 * sz;
  for (size_t i = 0; i < remain; ++i) {
    dst[idx + i] += src[idx + i] * c;
  }
}

template <typename T, typename src_aligned, typename dst_aligned>
inline void reduce(const typename T::value_type *src,
                   std::size_t size,
                   typename T::value_type *dst) {
  auto sz     = T::unroll_size;
  auto sz4    = T::unroll_size * 4;
  auto n4     = size / sz4;
  auto n1     = (size % sz4) / sz;
  auto remain = size % sz;
  for (size_t i = 0; i < n4; ++i) {
    auto d0 = T::template load<dst_aligned>(&dst[i * sz4 + sz * 0]);
    auto d1 = T::template load<dst_aligned>(&dst[i * sz4 + sz * 1]);
    auto d2 = T::template load<dst_aligned>(&dst[i * sz4 + sz * 2]);
    auto d3 = T::template load<dst_aligned>(&dst[i * sz4 + sz * 3]);
    auto s0 = T::template load<src_aligned>(&src[i * sz4 + sz * 0]);
    auto s1 = T::template load<src_aligned>(&src[i * sz4 + sz * 1]);
    auto s2 = T::template load<src_aligned>(&src[i * sz4 + sz * 2]);
    auto s3 = T::template load<src_aligned>(&src[i * sz4 + sz * 3]);
    d0      = T::add(s0, d0);
    d1      = T::add(s1, d1);
    d2      = T::add(s2, d2);
    d3      = T::add(s3, d3);
    T::template store<dst_aligned>(&dst[i * sz4 + sz * 0], d0);
    T::template store<dst_aligned>(&dst[i * sz4 + sz * 1], d1);
    T::template store<dst_aligned>(&dst[i * sz4 + sz * 2], d2);
    T::template store<dst_aligned>(&dst[i * sz4 + sz * 3], d3);
  }
  size_t idx = n4 * sz4;
  for (size_t i = 0; i < n1; ++i) {
    auto d = T::template load<dst_aligned>(&dst[idx + i * sz]);
    auto s = T::template load<src_aligned>(&src[idx + i * sz]);
    d      = T::add(s, d);
    T::template store<dst_aligned>(&dst[idx + i * sz], d);
  }
  idx += n1 * sz;
  for (size_t i = 0; i < remain; ++i) {
    dst[idx + i] += src[idx + i];
  }
}

template <typename T>
void fill(T *dst, size_t size, T value) {
  std::fill(dst, dst + size, value);
}

#if defined(CNN_USE_AVX)
#ifdef CNN_USE_DOUBLE
#define VECTORIZE_TYPE detail::double_avx
#else
#define VECTORIZE_TYPE detail::float_avx
#endif
#elif defined(CNN_USE_SSE)
#ifdef CNN_USE_DOUBLE
#define VECTORIZE_TYPE detail::double_sse
#else
#define VECTORIZE_TYPE detail::float_sse
#endif
#else
#ifdef CNN_USE_DOUBLE
#define VECTORIZE_TYPE detail::scalar_generic<double>
#else
#define VECTORIZE_TYPE detail::scalar_generic<float>
#endif
#endif

}  // namespace detail

// dst[i] += c
template <typename T>
void add(T c, std::size_t size, T *dst) {
  bool is_dst_aligned =
    VECTORIZE_TYPE::is_aligned((VECTORIZE_TYPE::value_type *)dst);
  if (is_dst_aligned) {
    detail::add<VECTORIZE_TYPE, std::true_type>(c, size, dst);
  } else {
    detail::add<VECTORIZE_TYPE, std::false_type>(c, size, dst);
  }
}

// dst[i] += src[i]
template <typename T>
void add(const T *src, std::size_t size, T *dst) {
  bool src_aligned =
    VECTORIZE_TYPE::is_aligned((VECTORIZE_TYPE::value_type *)src);
  bool dst_aligned =
    VECTORIZE_TYPE::is_aligned((VECTORIZE_TYPE::value_type *)dst);
  if (src_aligned) {
    if (dst_aligned) {
      detail::add<VECTORIZE_TYPE, std::true_type, std::true_type>(src, size,
                                                                  dst);
    } else {
      detail::add<VECTORIZE_TYPE, std::true_type, std::false_type>(src, size,
                                                                   dst);
    }
  } else {
    if (dst_aligned) {
      detail::add<VECTORIZE_TYPE, std::false_type, std::true_type>(src, size,
                                                                   dst);
    } else {
      detail::add<VECTORIZE_TYPE, std::false_type, std::false_type>(src, size,
                                                                    dst);
    }
  }
}

// dst[i] += c * src[i]
template <typename T>
void muladd(const T *src, T c, std::size_t size, T *dst) {
  bool src_aligned =
    VECTORIZE_TYPE::is_aligned((VECTORIZE_TYPE::value_type *)src);
  bool dst_aligned =
    VECTORIZE_TYPE::is_aligned((VECTORIZE_TYPE::value_type *)dst);
  if (src_aligned) {
    if (dst_aligned) {
      detail::muladd<VECTORIZE_TYPE, std::true_type, std::true_type>(src, c,
                                                                     size, dst);
    } else {
      detail::muladd<VECTORIZE_TYPE, std::true_type, std::false_type>(
        src, c, size, dst);
    }
  } else {
    if (dst_aligned) {
      detail::muladd<VECTORIZE_TYPE, std::false_type, std::true_type>(
        src, c, size, dst);
    } else {
      detail::muladd<VECTORIZE_TYPE, std::false_type, std::false_type>(
        src, c, size, dst);
    }
  }
}

// sum(s1[i] * s2[i])
template <typename T>
T dot(const T *s1, const T *s2, std::size_t size) {
  bool s1_aligned =
    VECTORIZE_TYPE::is_aligned((VECTORIZE_TYPE::value_type *)s1);
  bool s2_aligned =
    VECTORIZE_TYPE::is_aligned((VECTORIZE_TYPE::value_type *)s2);
  if (s1_aligned) {
    if (s2_aligned) {
      return detail::dot_product<VECTORIZE_TYPE, std::true_type,
                                 std::true_type>(s1, s2, size);
    } else {
      return detail::dot_product<VECTORIZE_TYPE, std::true_type,
                                 std::false_type>(s1, s2, size);
    }
  } else {
    if (s2_aligned) {
      return detail::dot_product<VECTORIZE_TYPE, std::false_type,
                                 std::true_type>(s1, s2, size);
    } else {
      return detail::dot_product<VECTORIZE_TYPE, std::false_type,
                                 std::false_type>(s1, s2, size);
    }
  }
}

/// dst[i] += src[i]
template <typename T>
void reduce(const T *src, std::size_t size, T *dst) {
  bool src_aligned =
    VECTORIZE_TYPE::is_aligned((VECTORIZE_TYPE::value_type *)src);
  bool dst_aligned =
    VECTORIZE_TYPE::is_aligned((VECTORIZE_TYPE::value_type *)dst);
  if (src_aligned) {
    if (dst_aligned) {
      detail::reduce<VECTORIZE_TYPE, std::true_type, std::true_type>(src, size,
                                                                     dst);
    } else {
      detail::reduce<VECTORIZE_TYPE, std::true_type, std::false_type>(src, size,
                                                                      dst);
    }
  } else {
    if (dst_aligned) {
      detail::reduce<VECTORIZE_TYPE, std::false_type, std::true_type>(src, size,
                                                                      dst);
    } else {
      detail::reduce<VECTORIZE_TYPE, std::false_type, std::false_type>(
        src, size, dst);
    }
  }
}

template <typename T>
inline void fill(T *dst, std::size_t size, T value) {
#if defined(_MSC_VER)
#if defined(_M_AMD64)

// On recent x86/x64 processors, REP string instructions are faster
// than SSE/AVX store instructions.
// CPUID feature flag (ERMSB) checking omitted
#if defined(CNN_USE_DOUBLE)
  union {
    unsigned __int64 dat;
    T value;
  } u;
  static_assert(sizeof(T) == sizeof(unsigned __int64), "size mismatch.");
  u.value = value;
  __stosq((unsigned __int64 *)dst, u.dat, size);
#else   // #if defined(CNN_USE_DOUBLE)
  union {
    unsigned long dat;
    T value;
  } u;
  static_assert(sizeof(T) == sizeof(unsigned long), "size mismatch.");
  u.value = value;
  __stosd((unsigned long *)dst, u.dat, size);
#endif  // #if defined(CNN_USE_DOUBLE)

#elif defined(_M_IX86)

#if defined(CNN_USE_DOUBLE)
  detail::fill(dst, size, value);
#else   // #if defined(CNN_USE_DOUBLE)
  union {
    unsigned long dat;
    T value;
  } u;
  static_assert(sizeof(T) == sizeof(unsigned long), "size mismatch.");
  u.value = value;
  __stosd((unsigned long *)dst, u.dat, size);
#endif  // #if defined(CNN_USE_DOUBLE)

#else  // !x86 && !x64
  detail::fill(dst, size, value);
#endif

#else  // #if defined(_MSC_VER)
  detail::fill(dst, size, value);
#endif
}

}  // namespace vectorize
