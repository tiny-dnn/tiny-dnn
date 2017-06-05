/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#if defined(CNN_USE_SSE) || defined(CNN_USE_AVX)
#include <immintrin.h>
#include "tiny_dnn/core/kernels/avx_kernel_common.h"
#endif
#include <cassert>
#include <cstdint>
#include <numeric>

namespace vectorize {
namespace detail {

// clang-format off

template <typename T> T set1(float x);
template <typename T> T set1(double x);
template <> CNN_MUSTINLINE float   set1<float >(float x)    { return x; }
template <> CNN_MUSTINLINE double  set1<double>(double x)   { return x; }

template <typename T> T loada(const float *px)                       { return *px; }
template <typename T> T loada(const double *px)                      { return *px; }

template <typename T> T loadu(const float *px);
template <typename T> T loadu(const double *px);
template <> CNN_MUSTINLINE float loadu<float>(const float *px)       { return *px; }
template <> CNN_MUSTINLINE double loadu<double>(const double *px)    { return *px; }

CNN_MUSTINLINE void storea(float *px, const float& x)     { *px = x; }
CNN_MUSTINLINE void storea(double *px, const double& x)   { *px = x; }

CNN_MUSTINLINE void storeu(float *px, const float &x)     { *px = x; }
CNN_MUSTINLINE void storeu(double *px, const double &x)   { *px = x; }

template <typename T> T load1(const T *px)                { return *px; }

CNN_MUSTINLINE void store1(float *px, const float &x)    { *px = x; }
CNN_MUSTINLINE void store1(double *px, const double &x)  { *px = x; }

template <typename T> T add1(const T &v1, const T &v2)            { return v1 + v2; }

template <typename T> T sub1(const T &v1, const T &v2)            { return v1 + v2; }

template <typename T> T mul1(const T &v1, const T &v2)            { return v1 * v2; }

template <typename T> T div1(const T &v1, const T &v2)            { return v1 * v2; }

template <typename T> T madd1(T v1, T v2, T v3)                   { return v1 * v2 + v3; }

template <typename T> T add(const T &v1, const T &v2)            { return v1 + v2; }

template <typename T> T sub(const T &v1, const T &v2)            { return v1 + v2; }

template <typename T> T mul(const T &v1, const T &v2)            { return v1 * v2; }

template <typename T> T div(const T &v1, const T &v2)            { return v1 * v2; }

template <typename T> T madd(T v1, T v2, T v3)                   { return v1 * v2 + v3; }

template <typename T> T hsum(const T &x)      { return x; }

CNN_MUSTINLINE float cvts(float v)    { return v; }
CNN_MUSTINLINE double cvts(double v)  { return v; }

#if defined(CNN_USE_SSE) || defined(CNN_USE_AVX)

template <> CNN_MUSTINLINE __m128  set1<__m128 >(float x)   { return _mm_set1_ps(x); }
template <> CNN_MUSTINLINE __m128d set1<__m128d>(double x)  { return _mm_set1_pd(x); }

template <> CNN_MUSTINLINE __m128  loada<__m128 >(const float  *px)  { return _mm_load_ps(px); }
template <> CNN_MUSTINLINE __m128d loada<__m128d>(const double *px)  { return _mm_load_pd(px); }

template <> CNN_MUSTINLINE __m128  loadu<__m128 >(const float  *px)  { return _mm_loadu_ps(px); }
template <> CNN_MUSTINLINE __m128d loadu<__m128d>(const double *px)  { return _mm_loadu_pd(px); }

CNN_MUSTINLINE void storea(float  *px, const __m128& x)   { _mm_store_ps(px, x); }
CNN_MUSTINLINE void storea(double *px, const __m128d& x)  { _mm_store_pd(px, x); }

CNN_MUSTINLINE void storeu(float  *px, const __m128 &x)   { _mm_storeu_ps(px, x); }
CNN_MUSTINLINE void storeu(double *px, const __m128d &x)  { _mm_storeu_pd(px, x); }

CNN_MUSTINLINE __m128  load1(const float  *px)            { return _mm_load_ss(px); }
CNN_MUSTINLINE __m128d load1(const double *px)            { return _mm_load_sd(px); }

CNN_MUSTINLINE void store1(float  *px, const __m128 &x)  { _mm_store_ss(px, x); }
CNN_MUSTINLINE void store1(double *px, const __m128d &x) { _mm_store_sd(px, x); }

CNN_MUSTINLINE __m128  add1(const __m128  &v1, const __m128  &v2) { return _mm_add_ss(v1, v2); }
CNN_MUSTINLINE __m128d add1(const __m128d &v1, const __m128d &v2) { return _mm_add_sd(v1, v2); }

CNN_MUSTINLINE __m128  sub1(const __m128  &v1, const __m128  &v2) { return _mm_sub_ss(v1, v2); }
CNN_MUSTINLINE __m128d sub1(const __m128d &v1, const __m128d &v2) { return _mm_sub_sd(v1, v2); }

CNN_MUSTINLINE __m128  mul1(const __m128  &v1, const __m128  &v2) { return _mm_mul_ss(v1, v2); }
CNN_MUSTINLINE __m128d mul1(const __m128d &v1, const __m128d &v2) { return _mm_mul_sd(v1, v2); }

CNN_MUSTINLINE __m128  div1(const __m128  &v1, const __m128  &v2) { return _mm_div_ss(v1, v2); }
CNN_MUSTINLINE __m128d div1(const __m128d &v1, const __m128d &v2) { return _mm_div_sd(v1, v2); }

CNN_MUSTINLINE __m128  madd1(const __m128  &v1, const __m128  &v2, const __m128  &v3) { return madd128_ss(v1, v2, v3); }
CNN_MUSTINLINE __m128d madd1(const __m128d &v1, const __m128d &v2, const __m128d &v3) { return madd128_sd(v1, v2, v3); }

CNN_MUSTINLINE __m128  add(const __m128  &v1, const __m128  &v2) { return _mm_add_ps(v1, v2); }
CNN_MUSTINLINE __m128d add(const __m128d &v1, const __m128d &v2) { return _mm_add_pd(v1, v2); }

CNN_MUSTINLINE __m128  sub(const __m128  &v1, const __m128  &v2) { return _mm_sub_ps(v1, v2); }
CNN_MUSTINLINE __m128d sub(const __m128d &v1, const __m128d &v2) { return _mm_sub_pd(v1, v2); }

CNN_MUSTINLINE __m128  mul(const __m128  &v1, const __m128  &v2) { return _mm_mul_ps(v1, v2); }
CNN_MUSTINLINE __m128d mul(const __m128d &v1, const __m128d &v2) { return _mm_mul_pd(v1, v2); }

CNN_MUSTINLINE __m128  div(const __m128  &v1, const __m128  &v2) { return _mm_div_ps(v1, v2); }
CNN_MUSTINLINE __m128d div(const __m128d &v1, const __m128d &v2) { return _mm_div_pd(v1, v2); }

CNN_MUSTINLINE __m128  madd(const __m128  &v1, const __m128  &v2, const __m128  &v3) { return madd128_ps(v1, v2, v3); }
CNN_MUSTINLINE __m128d madd(const __m128d &v1, const __m128d &v2, const __m128d &v3) { return madd128_pd(v1, v2, v3); }

CNN_MUSTINLINE __m128  hsum(const __m128  &x) { return hsum128_ps(x); }
CNN_MUSTINLINE __m128d hsum(const __m128d &x) { return hsum128_pd(x); }

CNN_MUSTINLINE float cvts(const __m128 &v)    { return _mm_cvtss_f32(v); }
CNN_MUSTINLINE double cvts(const __m128d &v)  { return _mm_cvtsd_f64(v); }

#endif // #if defined(CNN_USE_SSE) || defined(CNN_USE_AVX)

#ifdef CNN_USE_AVX

template <> CNN_MUSTINLINE __m256  set1<__m256 >(float x)   { return _mm256_set1_ps(x); }
template <> CNN_MUSTINLINE __m256d set1<__m256d>(double x)  { return _mm256_set1_pd(x); }

template <> CNN_MUSTINLINE __m256  loada<__m256 >(const float  *px)  { return _mm256_load_ps(px); }
template <> CNN_MUSTINLINE __m256d loada<__m256d>(const double *px)  { return _mm256_load_pd(px); }

template <> CNN_MUSTINLINE __m256  loadu<__m256 >(const float  *px)  { return _mm256_loadu_ps(px); }
template <> CNN_MUSTINLINE __m256d loadu<__m256d>(const double *px)  { return _mm256_loadu_pd(px); }

CNN_MUSTINLINE void storea(float  *px, const __m256& x)   { _mm256_store_ps(px, x); }
CNN_MUSTINLINE void storea(double *px, const __m256d& x)  { _mm256_store_pd(px, x); }

CNN_MUSTINLINE void storeu(float  *px, const __m256 &x)   { _mm256_storeu_ps(px, x); }
CNN_MUSTINLINE void storeu(double *px, const __m256d &x)  { _mm256_storeu_pd(px, x); }

CNN_MUSTINLINE __m256  add(const __m256  &v1, const __m256  &v2) { return _mm256_add_ps(v1, v2); }
CNN_MUSTINLINE __m256d add(const __m256d &v1, const __m256d &v2) { return _mm256_add_pd(v1, v2); }

CNN_MUSTINLINE __m256  sub(const __m256  &v1, const __m256  &v2) { return _mm256_sub_ps(v1, v2); }
CNN_MUSTINLINE __m256d sub(const __m256d &v1, const __m256d &v2) { return _mm256_sub_pd(v1, v2); }

CNN_MUSTINLINE __m256  mul(const __m256  &v1, const __m256  &v2) { return _mm256_mul_ps(v1, v2); }
CNN_MUSTINLINE __m256d mul(const __m256d &v1, const __m256d &v2) { return _mm256_mul_pd(v1, v2); }

CNN_MUSTINLINE __m256  div(const __m256  &v1, const __m256  &v2) { return _mm256_div_ps(v1, v2); }
CNN_MUSTINLINE __m256d div(const __m256d &v1, const __m256d &v2) { return _mm256_div_pd(v1, v2); }

CNN_MUSTINLINE __m256  madd(const __m256  &v1, const __m256  &v2, const __m256  &v3) { return madd256_ps(v1, v2, v3); }
CNN_MUSTINLINE __m256d madd(const __m256d &v1, const __m256d &v2, const __m256d &v3) { return madd256_pd(v1, v2, v3); }

CNN_MUSTINLINE __m128  hsum(const __m256  &x) { return hsum256_ps(x); }
CNN_MUSTINLINE __m128d hsum(const __m256d &x) { return hsum256_pd(x); }

#endif // #ifdef CNN_USE_AVX

// clang-format on

template <typename ElementT>
struct type_holder {
  typedef ElementT element_type;
  typedef ElementT register_type;
  enum { unroll_size = 1 };
  static register_type zero() { return 0; }
};

#if defined(CNN_USE_AVX)

template <>
struct type_holder<float> {
  typedef float element_type;
  typedef __m256 register_type;
  enum { unroll_size = 8 };
  static register_type zero() { return _mm256_setzero_ps(); }
};

template <>
struct type_holder<double> {
  typedef double element_type;
  typedef __m256d register_type;
  enum { unroll_size = 4 };
  static register_type zero() { return _mm256_setzero_pd(); }
};

#elif defined(CNN_USE_SSE)

template <>
struct type_holder<float> {
  typedef float element_type;
  typedef __m128 register_type;
  enum { unroll_size = 4 };
  static register_type zero() { return _mm_setzero_ps(); }
};

template <>
struct type_holder<double> {
  typedef double element_type;
  typedef __m128d register_type;
  enum { unroll_size = 2 };
  static register_type zero() { return _mm_setzero_pd(); }
};

#endif

// generic dot-product
template <typename T>
inline T dot_product(const T *f1, const T *f2, size_t size) {
  typedef typename type_holder<T>::register_type register_type;
  register_type zero = type_holder<T>::zero();
  register_type r0 = zero;
  register_type r1 = zero;
  register_type r2 = zero;
  register_type r3 = zero;
  auto sz          = type_holder<T>::unroll_size;
  auto sz4         = type_holder<T>::unroll_size * 4;
  auto n4          = size / sz4;
  auto n1          = (size % sz4) / sz;
  auto remain      = size % sz;
  for (size_t i = 0; i < n4; ++i) {
    auto s10 = loadu<register_type>(&f1[i * sz4 + sz * 0]);
    auto s11 = loadu<register_type>(&f1[i * sz4 + sz * 1]);
    auto s12 = loadu<register_type>(&f1[i * sz4 + sz * 2]);
    auto s13 = loadu<register_type>(&f1[i * sz4 + sz * 3]);
    auto s20 = loadu<register_type>(&f2[i * sz4 + sz * 0]);
    auto s21 = loadu<register_type>(&f2[i * sz4 + sz * 1]);
    auto s22 = loadu<register_type>(&f2[i * sz4 + sz * 2]);
    auto s23 = loadu<register_type>(&f2[i * sz4 + sz * 3]);
    r0       = madd(s10, s20, r0);
    r1       = madd(s11, s21, r1);
    r2       = madd(s12, s22, r2);
    r3       = madd(s13, s23, r3);
  }
  size_t idx = n4 * sz4;
  for (size_t i = 0; i < n1; ++i) {
    auto s1 = loadu<register_type>(&f1[idx + i * sz]);
    auto s2 = loadu<register_type>(&f2[idx + i * sz]);
    r0      = madd(s1, s2, r0);
  }
  r0    = add(r0, r1);
  r2    = add(r2, r3);
  r0    = add(r0, r2);
  T sum = cvts(hsum(r0));
  idx += n1 * sz;
  for (size_t i = 0; i < remain; ++i) {
    sum += f1[idx + i] * f2[idx + i];
  }
  return sum;
}

template <typename T>
inline void add(T c, size_t size, T *dst) {
  typedef typename type_holder<T>::register_type register_type;
  register_type c2 = set1<register_type>(c);
  auto sz          = type_holder<T>::unroll_size;
  auto sz4         = type_holder<T>::unroll_size * 4;
  auto n4          = size / sz4;
  auto n1          = (size % sz4) / sz;
  auto remain      = size % sz;
  for (size_t i = 0; i < n4; ++i) {
    auto d0 = loadu<register_type>(&dst[i * sz4 + sz * 0]);
    auto d1 = loadu<register_type>(&dst[i * sz4 + sz * 1]);
    auto d2 = loadu<register_type>(&dst[i * sz4 + sz * 2]);
    auto d3 = loadu<register_type>(&dst[i * sz4 + sz * 3]);
    d0      = add(c2, d0);
    d1      = add(c2, d1);
    d2      = add(c2, d2);
    d3      = add(c2, d3);
    storeu(&dst[i * sz4 + sz * 0], d0);
    storeu(&dst[i * sz4 + sz * 1], d1);
    storeu(&dst[i * sz4 + sz * 2], d2);
    storeu(&dst[i * sz4 + sz * 3], d3);
  }
  size_t idx = n4 * sz4;
  for (size_t i = 0; i < n1; ++i) {
    auto d = loadu<register_type>(&dst[idx + i * sz]);
    d      = add(c2, d);
    storeu(&dst[idx + i * sz], d);
  }
  idx += n1 * sz;
  for (size_t i = 0; i < remain; ++i) {
    dst[idx + i] += c;
  }
}

template <typename T>
inline void add(const T *src, size_t size, T *dst) {
  typedef typename type_holder<T>::register_type register_type;
  auto sz     = type_holder<T>::unroll_size;
  auto sz4    = type_holder<T>::unroll_size * 4;
  auto n4     = size / sz4;
  auto n1     = (size % sz4) / sz;
  auto remain = size % sz;
  for (size_t i = 0; i < n4; ++i) {
    auto d0 = loadu<register_type>(&dst[i * sz4 + sz * 0]);
    auto d1 = loadu<register_type>(&dst[i * sz4 + sz * 1]);
    auto d2 = loadu<register_type>(&dst[i * sz4 + sz * 2]);
    auto d3 = loadu<register_type>(&dst[i * sz4 + sz * 3]);
    auto s0 = loadu<register_type>(&src[i * sz4 + sz * 0]);
    auto s1 = loadu<register_type>(&src[i * sz4 + sz * 1]);
    auto s2 = loadu<register_type>(&src[i * sz4 + sz * 2]);
    auto s3 = loadu<register_type>(&src[i * sz4 + sz * 3]);
    d0      = add(s0, d0);
    d1      = add(s1, d1);
    d2      = add(s2, d2);
    d3      = add(s3, d3);
    storeu(&dst[i * sz4 + sz * 0], d0);
    storeu(&dst[i * sz4 + sz * 1], d1);
    storeu(&dst[i * sz4 + sz * 2], d2);
    storeu(&dst[i * sz4 + sz * 3], d3);
  }
  size_t idx = n4 * sz4;
  for (size_t i = 0; i < n1; ++i) {
    auto d = loadu<register_type>(&dst[idx + i * sz]);
    auto s = loadu<register_type>(&src[idx + i * sz]);
    d      = add(s, d);
    storeu(&dst[idx + i * sz], d);
  }
  idx += n1 * sz;
  for (size_t i = 0; i < remain; ++i) {
    dst[idx + i] += src[idx + i];
  }
}

template <typename T>
inline void muladd(const T *src, T c, size_t size, T *dst) {
  typedef typename type_holder<T>::register_type register_type;
  auto factor = set1<register_type>(c);
  auto sz     = type_holder<T>::unroll_size;
  auto sz4    = type_holder<T>::unroll_size * 4;
  auto n4     = size / sz4;
  auto n1     = (size % sz4) / sz;
  auto remain = size % sz;
  for (size_t i = 0; i < n4; ++i) {
    auto d0 = loadu<register_type>(&dst[i * sz4 + sz * 0]);
    auto d1 = loadu<register_type>(&dst[i * sz4 + sz * 1]);
    auto d2 = loadu<register_type>(&dst[i * sz4 + sz * 2]);
    auto d3 = loadu<register_type>(&dst[i * sz4 + sz * 3]);
    auto s0 = loadu<register_type>(&src[i * sz4 + sz * 0]);
    auto s1 = loadu<register_type>(&src[i * sz4 + sz * 1]);
    auto s2 = loadu<register_type>(&src[i * sz4 + sz * 2]);
    auto s3 = loadu<register_type>(&src[i * sz4 + sz * 3]);
    d0      = madd(s0, factor, d0);
    d1      = madd(s1, factor, d1);
    d2      = madd(s2, factor, d2);
    d3      = madd(s3, factor, d3);
    storeu(&dst[i * sz4 + sz * 0], d0);
    storeu(&dst[i * sz4 + sz * 1], d1);
    storeu(&dst[i * sz4 + sz * 2], d2);
    storeu(&dst[i * sz4 + sz * 3], d3);
  }
  size_t idx = n4 * sz4;
  for (size_t i = 0; i < n1; ++i) {
    auto d = loadu<register_type>(&dst[idx + i * sz]);
    auto s = loadu<register_type>(&src[idx + i * sz]);
    d      = madd(s, factor, d);
    storeu(&dst[idx + i * sz], d);
  }
  idx += n1 * sz;
  for (size_t i = 0; i < remain; ++i) {
    dst[idx + i] += src[idx + i] * c;
  }
}

template <typename T>
inline void reduce(const T *src, size_t size, T *dst) {
  typedef typename type_holder<T>::register_type register_type;
  auto sz     = type_holder<T>::unroll_size;
  auto sz4    = type_holder<T>::unroll_size * 4;
  auto n4     = size / sz4;
  auto n1     = (size % sz4) / sz;
  auto remain = size % sz;
  for (size_t i = 0; i < n4; ++i) {
    auto d0 = loadu<register_type>(&dst[i * sz4 + sz * 0]);
    auto d1 = loadu<register_type>(&dst[i * sz4 + sz * 1]);
    auto d2 = loadu<register_type>(&dst[i * sz4 + sz * 2]);
    auto d3 = loadu<register_type>(&dst[i * sz4 + sz * 3]);
    auto s0 = loadu<register_type>(&src[i * sz4 + sz * 0]);
    auto s1 = loadu<register_type>(&src[i * sz4 + sz * 1]);
    auto s2 = loadu<register_type>(&src[i * sz4 + sz * 2]);
    auto s3 = loadu<register_type>(&src[i * sz4 + sz * 3]);
    d0      = add(s0, d0);
    d1      = add(s1, d1);
    d2      = add(s2, d2);
    d3      = add(s3, d3);
    storeu(&dst[i * sz4 + sz * 0], d0);
    storeu(&dst[i * sz4 + sz * 1], d1);
    storeu(&dst[i * sz4 + sz * 2], d2);
    storeu(&dst[i * sz4 + sz * 3], d3);
  }
  size_t idx = n4 * sz4;
  for (size_t i = 0; i < n1; ++i) {
    auto d = loadu<register_type>(&dst[idx + i * sz]);
    auto s = loadu<register_type>(&src[idx + i * sz]);
    d      = add(s, d);
    storeu(&dst[idx + i * sz], d);
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

}  // namespace detail

// dst[i] += c
template <typename T>
void add(T c, size_t size, T *dst) {
  detail::add(c, size, dst);
}

// dst[i] += src[i]
template <typename T>
void add(const T *src, size_t size, T *dst) {
  detail::add(src, size, dst);
}

// dst[i] += c * src[i]
template <typename T>
void muladd(const T *src, T c, size_t size, T *dst) {
  detail::muladd(src, c, size, dst);
}

// sum(s1[i] * s2[i])
template <typename T>
T dot(const T *s1, const T *s2, size_t size) {
  return detail::dot_product(s1, s2, size);
}

/// dst[i] += src[i]
template <typename T>
void reduce(const T *src, size_t size, T *dst) {
  detail::reduce(src, size, dst);
}

template <typename T>
inline void fill(T *dst, size_t size, T value) {
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
