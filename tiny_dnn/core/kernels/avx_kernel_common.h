/*
    Copyright (c) 2016, Taiga Nomi, Edgar Riba
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#ifndef CNN_USE_AVX
#error Advanced Vector Extensions required.
#endif

#ifndef _mm256_set_m128
#define _mm256_set_m128(va, vb) \
  _mm256_insertf128_ps(_mm256_castps128_ps256(vb), va, 1)
#endif

#ifdef CNN_USE_AVX2
inline __m256 madd256_ps(__m256 a, __m256 b, __m256 c) {
  return _mm256_fmadd_ps(a, b, c);
}
inline __m128 madd128_ps(__m128 a, __m128 b, __m128 c) {
  return _mm_fmadd_ps(a, b, c);
}
inline __m128 madd128_ss(__m128 a, __m128 b, __m128 c) {
  return _mm_fmadd_ss(a, b, c);
}
inline __m256d madd256_pd(__m256d a, __m256d b, __m256d c) {
  return _mm256_fmadd_pd(a, b, c);
}
inline __m128d madd128_pd(__m128d a, __m128d b, __m128d c) {
  return _mm_fmadd_pd(a, b, c);
}
inline __m128d madd128_sd(__m128d a, __m128d b, __m128d c) {
  return _mm_fmadd_sd(a, b, c);
}
#else
inline __m256 madd256_ps(__m256 a, __m256 b, __m256 c) {
  return _mm256_add_ps(_mm256_mul_ps(a, b), c);
}
inline __m128 madd128_ps(__m128 a, __m128 b, __m128 c) {
  return _mm_add_ps(_mm_mul_ps(a, b), c);
}
inline __m128 madd128_ss(__m128 a, __m128 b, __m128 c) {
  return _mm_add_ss(_mm_mul_ss(a, b), c);
}
inline __m256d madd256_pd(__m256d a, __m256d b, __m256d c) {
  return _mm256_add_pd(_mm256_mul_pd(a, b), c);
}
inline __m128d madd128_pd(__m128d a, __m128d b, __m128d c) {
  return _mm_add_pd(_mm_mul_pd(a, b), c);
}
inline __m128d madd128_sd(__m128d a, __m128d b, __m128d c) {
  return _mm_add_sd(_mm_mul_sd(a, b), c);
}
#endif

// Horizontally add elements of __m256 type argument (sadly, _mm256_hadd_ps
// isn't good enough)
// http://stackoverflow.com/a/13222410/4699324
// x = ( x7, x6, x5, x4, x3, x2, x1, x0 )
// use _mm_cvtss_f32 if you need a float result instead of a __m128
inline __m128 hsum256_ps(__m256 x) {
  // hiQuad = ( x7, x6, x5, x4 )
  const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
  // loQuad = ( x3, x2, x1, x0 )
  const __m128 loQuad = _mm256_castps256_ps128(x);
  // sumQuad = ( x3+x7, x2+x6, x1+x5, x0+x4 )
  const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
  // loDual = ( -, -, x1+x5, x0+x4 )
  const __m128 loDual = sumQuad;
  // hiDual = ( -, -, x3+x7, x2+x6 )
  const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
  // sumDual = ( -, -, x1+x3 + x5+x7, x0+x2 + x4+x6 )
  const __m128 sumDual = _mm_add_ps(loDual, hiDual);
  // lo = ( -, -, -, x0+x2 + x4+x6 )
  const __m128 lo = sumDual;
  // hi = ( -, -, -, x1+x3 + x5+x7 )
  const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
  // sum = ( -, -, -, x0+x1+x2+x3 + x4+x5+x6+x7 )
  const __m128 sum = _mm_add_ss(lo, hi);
  return sum;
}

// Horizontally add elements of each __m256 type arguments at once
inline __m128 hsum2x256_ps(__m256 a, __m256 b) {
  // (b3, b2, b1, b0, a3, a2, a1, a0)
  __m256 x = _mm256_permute2f128_ps(a, b, 0x20);
  // (b7, b6, b5, b4, a7, a6, a5, a4)
  __m256 y = _mm256_permute2f128_ps(a, b, 0x31);
  // (b3+b7, b2+b6, b1+b5, b0+b4, a3+a7, a2+a6, a1+a5, a0+a4)
  x = _mm256_add_ps(x, y);
  // (-, -, b3+b7, b2+b6, -, -, a3+a7, a2+a6)
  y = _mm256_permute_ps(x, _MM_SHUFFLE(3, 2, 3, 2));
  // (-, -, b1+b5+b3+b7, b0+b4+b2+b6, -, -, a1+a5+a3+a7, a0+a4+a2+a6)
  x = _mm256_add_ps(x, y);
  // (-, -, -, b1+b5+b3+b7, -, -, -, a1+a5+a3+a7)
  y = _mm256_permute_ps(x, _MM_SHUFFLE(1, 1, 1, 1));
  // (-, -, -, b1+b5+b3+b7+b0+b4+b2+b6, -, -, -, a1+a5+a3+a7+a0+a4+a2+a6)
  x = _mm256_add_ps(x, y);
  // (-, -, -, b1+b5+b3+b7+b0+b4+b2+b6)
  __m128 upper = _mm256_extractf128_ps(x, 1);
  // (-, -, -, -, -, -, b1+b5+b3+b7+b0+b4+b2+b6, a1+a5+a3+a7+a0+a4+a2+a6)
  __m128 ret = _mm_unpacklo_ps(_mm256_castps256_ps128(x), upper);
  return ret;
}

// Horizontally add elements of each __m256 type arguments at once
// in a : ( a7, a6, a5, a4, a3, a2, a1, a0 )
// in b : ( b7, b6, b5, b4, b3, b2, b1, b0 )
// in c : ( c7, c6, c5, c4, c3, c2, c1, c0 )
// in d : ( d7, d6, d5, d4, d3, d2, d1, d0 )
// out  : ( dsum, csum, bsum, asum )
inline __m128 hsum4x256_ps(const __m256 &a,
                           const __m256 &b,
                           const __m256 &c,
                           const __m256 &d) {
  // (b3,b2,b1,b0, a3,a2,a1,a0)
  __m256 w = _mm256_permute2f128_ps(a, b, 0x20);
  // (b7,b6,b5,b4, a7,a6,a5,a4)
  __m256 x = _mm256_permute2f128_ps(a, b, 0x31);
  // (d3,d2,d1,d0, c3,c2,c1,c0)
  __m256 y = _mm256_permute2f128_ps(c, d, 0x20);
  // (d7,d6,d5,d4, c7,c6,c5,c4)
  __m256 z = _mm256_permute2f128_ps(c, d, 0x31);

  // (b3,b2,b1,b0, a3,a2,a1,a0)
  // (b7,b6,b5,b4, a7,a6,a5,a4)
  w = _mm256_add_ps(w, x);
  // (-,-,b3,b2, -,-,a3,a2)
  // (-,-,b7,b6, -,-,a7,a6)
  x = _mm256_permute_ps(w, _MM_SHUFFLE(3, 2, 3, 2));
  // (-,-,b1,b0, -,-,a1,a0)
  // (-,-,b5,b4, -,-,a5,a4)
  // (-,-,b3,b2, -,-,a3,a2)
  // (-,-,b7,b6, -,-,a7,a6)
  w = _mm256_add_ps(w, x);

  // (d3,d2,d1,d0, c3,c2,c1,c0)
  // (d7,d6,d5,d4, c7,c6,c5,c4)
  y = _mm256_add_ps(y, z);
  // (-,-,d3,d2, -,-,c3,c2)
  // (-,-,d7,d6, -,-,c7,c6)
  z = _mm256_permute_ps(y, _MM_SHUFFLE(3, 2, 3, 2));
  // (-,-,d1,d0, -,-,c1,c0)
  // (-,-,d5,d4, -,-,c5,c4)
  // (-,-,d3,d2, -,-,c3,c2)
  // (-,-,d7,d6, -,-,c7,c6)
  z = _mm256_add_ps(y, z);

  // d1,d0,b1,b0, c1,c0,a1,a0)
  // d5,d4,b5,b4, c5,c4,a5,a4)
  // d3,d2,b3,b2, c3,c2,a3,a2)
  // d7,d6,b7,b6, c7,c6,a7,a6)
  w = _mm256_castpd_ps(
    _mm256_unpacklo_pd(_mm256_castps_pd(w), _mm256_castps_pd(z)));

  // (d0,d1,b0,b1, c0,c1,a0,a1)
  // (d4,d5,b4,b5, c4,c5,a4,a5)
  // (d2,d3,b2,b3, c2,c3,a2,a3)
  // (d6,d7,b6,b7, c6,c7,a6,a7)
  x = _mm256_permute_ps(w, _MM_SHUFFLE(2, 3, 0, 1));

  // (d1,d1,b1,b1, c1,c1,a1,a1)
  // (d5,d5,b5,b5, c5,c5,a5,a5)
  // (d3,d3,b3,b3, c3,c3,a3,a3)
  // (d7,d7,b7,b7, c7,c7,a7,a7)
  // (d0,d0,b0,b0, c0,c0,a0,a0)
  // (d4,d4,b4,b4, c4,c4,a4,a4)
  // (d2,d2,b2,b2, c2,c2,a2,a2)
  // (d6,d6,b6,b6, c6,c6,a6,a6)
  w = _mm256_add_ps(w, x);

  // (d1,d1,b1,b1)
  // (d5,d5,b5,b5)
  // (d3,d3,b3,b3)
  // (d7,d7,b7,b7)
  // (d0,d0,b0,b0)
  // (d4,d4,b4,b4)
  // (d2,d2,b2,b2)
  // (d6,d6,b6,b6)
  __m128 upper = _mm256_extractf128_ps(w, 1);

  // (d1,c1,b1,a1)
  // (d5,c5,b5,a5)
  // (d3,c3,b3,a3)
  // (d7,c7,b7,a7)
  // (d0,c0,b0,a0)
  // (d4,c4,b4,a4)
  // (d2,c2,b2,a2)
  // (d6,c6,b6,a6)
  __m128 ret =
    _mm_blend_ps(_mm256_castps256_ps128(w), upper, 0x0A /* 0b1010 */);

  return ret;
}

inline __m128d hsum256_pd(__m256d x) {
  // hiDual = ( x3, x2 )
  const __m128d hiDual = _mm256_extractf128_pd(x, 1);
  // loDual = ( x1, x0 )
  const __m128d loDual = _mm256_castpd256_pd128(x);
  // sumQuad = ( x2+x3, x0+x1 )
  const __m128d sumDual = _mm_add_pd(loDual, hiDual);
  // sum = ( 0, x0+x1+x2+x3 );
  const __m128d sum = _mm_hadd_pd(sumDual, _mm_setzero_pd());
  return sum;
}

template <int n>
struct foobar : std::false_type {};

// Byte Shift YMM Register Across 128-bit Lanes

#ifdef CNN_USE_AVX2

template <bool>
struct Range;

template <unsigned int N, typename = Range<true>>
struct m256_shift_left_impl {};

template <unsigned int N>
struct m256_shift_left_impl<N, Range<N == 0>> {
  static __m256 doit(__m256i a) { return a; }
};

template <unsigned int N>
struct m256_shift_left_impl<N, Range<(0 < N && N < 16)>> {
  static __m256 doit(__m256 a) {
    __m256 mask = _mm256_permute2f128_ps(a, a, 0x08);
    return _mm256_castsi256_ps(_mm256_alignr_epi8(
      _mm256_castps_si256(a), _mm256_castps_si256(mask), 16 - N));
  }
};

template <unsigned int N>
inline __m256 leftShift(__m256 a) {
  return m256_shift_left_impl<N>::doit(a);
}

#else  // #ifdef CNN_USE_AVX2

// limitation : shift amount is immediate and is multiples of 4

template <int n>
inline __m256 leftShift(__m256 a) {
  static_assert(foobar<n>::value, "unsupported shift amount");
  return a;
}

// http://stackoverflow.com/q/19516585
template <>
inline __m256 leftShift<4>(__m256 x) {
  // x  = (x7, x6, x5, x4, x3, x2, x1, x0)

  // t0 = (x6, x5, x4, x7, x2, x1, x0, x3)
  __m256 t0 = _mm256_permute_ps(x, _MM_SHUFFLE(2, 1, 0, 3));
  // t1 = (x2, x1, x0, x3, 0, 0, 0, 0)
  __m256 t1 = _mm256_permute2f128_ps(t0, t0, 0x08);
  // y  = (x6, x5, x4, x3, x2, x1, x0, 0)
  __m256 y = _mm256_blend_ps(t0, t1, 0x11);
  return y;
}

// http://stackoverflow.com/q/19516585
template <>
inline __m256 leftShift<8>(__m256 x) {
  // x  = (x7, x6, x5, x4, x3, x2, x1, x0)

  // t0 = (x5, x4, x7, x6, x1, x0, x3, x2)
  __m256 t0 = _mm256_permute_ps(x, _MM_SHUFFLE(1, 0, 3, 2));
  // t1 = (x1, x0, x3, x2, 0, 0, 0, 0)
  __m256 t1 = _mm256_permute2f128_ps(t0, t0, 0x08);
  // y  = (x5, x4, x3, x2, x1, x0, 0, 0)
  __m256 y = _mm256_blend_ps(t0, t1, 0x33 /* 0b00110011 */);
  return y;
}

template <>
inline __m256 leftShift<12>(__m256 x) {
  // x  = (x7, x6, x5, x4, x3, x2, x1, x0)

  // t0 = (x4, x7, x6, x5, x0, x3, x2, x1)
  __m256 t0 = _mm256_permute_ps(x, _MM_SHUFFLE(0, 3, 2, 1));
  // t1 = (x0, x3, x2, x1, 0, 0, 0, 0)
  __m256 t1 = _mm256_permute2f128_ps(t0, t0, 0x08);
  // y  = (x4, x3, x2, x1, x0, 0, 0, 0)
  __m256 y = _mm256_blend_ps(t0, t1, 0x77 /* 0b01110111 */);
  return y;
}

template <>
inline __m256 leftShift<16>(__m256 x) {
  // x  = (x7, x6, x5, x4, x3, x2, x1, x0)

  // y  = (x3, x2, x1, x0, 0, 0, 0, 0)
  __m256 y = _mm256_permute2f128_ps(x, x, 0x08);
  return y;
}

template <>
inline __m256 leftShift<20>(__m256 x) {
  // x  = (x7, x6, x5, x4, x3, x2, x1, x0)

  // t0 = (x6, x5, x4, x7, x2, x1, x0, x3)
  __m256 t0 = _mm256_permute_ps(x, _MM_SHUFFLE(2, 1, 0, 3));
  // t1 = (x2, x1, x0, x3, 0, 0, 0, 0)
  __m256 t1 = _mm256_permute2f128_ps(t0, t0, 0x08);
  // y  = (x2, x1, x0, 0, 0, 0, 0, 0)
  __m256 y = _mm256_blend_ps(t1, _mm256_setzero_ps(), 0x10 /* 0b00010000 */);
  return y;
}

template <>
inline __m256 leftShift<24>(__m256 x) {
  // x  = (x7, x6, x5, x4, x3, x2, x1, x0)

  // t0 = (x5, x4, x7, x6, x1, x0, x3, x2)
  __m256 t0 = _mm256_permute_ps(x, _MM_SHUFFLE(1, 0, 3, 2));
  // t1 = (x1, x0, x3, x2, 0, 0, 0, 0)
  __m256 t1 = _mm256_permute2f128_ps(t0, t0, 0x08);
  // y  = (x1, x0, 0, 0, 0, 0, 0, 0)
  __m256 y = _mm256_blend_ps(_mm256_setzero_ps(), t1, 0xC0 /* 0b11000000 */);
  return y;
}

template <>
inline __m256 leftShift<28>(__m256 x) {
  // x  = (x7, x6, x5, x4, x3, x2, x1, x0)

  // t0 = (x4, x7, x6, x5, x0, x3, x2, x1)
  __m256 t0 = _mm256_permute_ps(x, _MM_SHUFFLE(0, 3, 2, 1));
  // t1 = (x0, x3, x2, x1, 0, 0, 0, 0)
  __m256 t1 = _mm256_permute2f128_ps(t0, t0, 0x08);
  // y  = (x0, 0, 0, 0, 0, 0, 0, 0)
  __m256 y = _mm256_blend_ps(_mm256_setzero_ps(), t1, 0x80 /* 0b10000000 */);
  return y;
}

#endif  // #ifdef CNN_USE_AVX2

template <int n>
inline __m256 rightShift(__m256 a) {
  static_assert(foobar<n>::value, "unsupported shift amount");
  return a;
}

// http://stackoverflow.com/a/19532415/4699324
template <>
inline __m256 rightShift<4>(__m256 x) {
  // x  = (x7, x6, x5, x4, x3, x2, x1, x0)

  // t0 = (x4, x7, x6, x5, x0, x3, x2, x1)
  __m256 t0 = _mm256_permute_ps(x, _MM_SHUFFLE(0, 3, 2, 1));
  // t1 = (0, 0, 0, 0, x4, x7, x6, x5)
  __m256 t1 = _mm256_permute2f128_ps(t0, t0, 0x81);

  //      ( -, x7, x6, x5,  -, x3, x2, x1)
  //      ( 0,  -,  -,  -, x4,  -,  -,  -)
  // y  = ( 0, x7, x6, x5, x4, x3, x2, x1)
  __m256 y = _mm256_blend_ps(t0, t1, 0x88 /* 0b10001000 */);
  return y;
}

template <>
inline __m256 rightShift<8>(__m256 x) {
  // x  = (x7, x6, x5, x4, x3, x2, x1, x0)

  // t0 = (x5, x4, x7, x6, x1, x0, x3, x2)
  __m256 t0 = _mm256_permute_ps(x, _MM_SHUFFLE(1, 0, 3, 2));
  // t1 = (0, 0, 0, 0, x5, x4, x7, x6)
  __m256 t1 = _mm256_permute2f128_ps(t0, t0, 0x81);

  //      ( -,  -, x7, x6,  -,  -, x3, x2)
  //      ( 0,  0,  -,  -, x5, x4,  -,  -)
  // y  = ( 0,  0, x7, x6, x5, x4, x3, x2)
  __m256 y = _mm256_blend_ps(t0, t1, 0xCC /* 0b11001100 */);
  return y;
}

template <>
inline __m256 rightShift<12>(__m256 x) {
  // x  = (x7, x6, x5, x4, x3, x2, x1, x0)

  // t0 = (x6, x5, x4, x7, x2, x1, x0, x3)
  __m256 t0 = _mm256_permute_ps(x, _MM_SHUFFLE(2, 1, 0, 3));
  // t1 = ( 0,  0,  0,  0, x6, x5, x4, x7)
  __m256 t1 = _mm256_permute2f128_ps(t0, t0, 0x81);

  //      ( -,  -,  -, x7,  -,  -,  -, x3)
  //      ( 0,  0,  0,  -, x6, x5, x4,  -)
  // y  = ( 0,  0,  0, x7, x6, x5, x4, x3)
  __m256 y = _mm256_blend_ps(t0, t1, 0xEE /* 0b11101110 */);
  return y;
}

template <>
inline __m256 rightShift<16>(__m256 x) {
  // x  = (x7, x6, x5, x4, x3, x2, x1, x0)

  // y  = ( 0,  0,  0,  0, x7, x6, x5, x4)
  __m256 y = _mm256_permute2f128_ps(x, x, 0x81);
  return y;
}

template <>
inline __m256 rightShift<20>(__m256 x) {
  // x  = (x7, x6, x5, x4, x3, x2, x1, x0)

  // t0 = (x4, x7, x6, x5, x0, x3, x2, x1)
  __m256 t0 = _mm256_permute_ps(x, _MM_SHUFFLE(0, 3, 2, 1));
  // t1 = ( 0,  0,  0,  0, x4, x7, x6, x5)
  __m256 t1 = _mm256_permute2f128_ps(t0, t0, 0x81);

  //      ( -,  -,  -,  -,  -, x7, x6, x5)
  //      ( 0,  0,  0,  0,  0,  -,  -,  -)
  // y  = ( 0,  0,  0,  0,  0, x7, x6, x5)
  __m256 y = _mm256_blend_ps(t1, _mm256_setzero_ps(), 0xF8 /* 0b11111000 */);
  return y;
}

template <>
inline __m256 rightShift<24>(__m256 x) {
  // x  = (x7, x6, x5, x4, x3, x2, x1, x0)

  // t0 = (x5, x4, x7, x6, x1, x0, x3, x2)
  __m256 t0 = _mm256_permute_ps(x, _MM_SHUFFLE(1, 0, 3, 2));
  // t1 = ( 0,  0,  0,  0, x5, x4, x7, x6)
  __m256 t1 = _mm256_permute2f128_ps(t0, t0, 0x81);

  //      ( -,  -,  -,  -,  -,  -, x7, x6)
  //      ( 0,  0,  0,  0,  0,  0,  -,  -)
  // y  = ( 0,  0,  0,  0,  0,  0, x7, x6)
  __m256 y = _mm256_blend_ps(t1, _mm256_setzero_ps(), 0xFC /* 0b11111100 */);
  return y;
}

template <>
inline __m256 rightShift<28>(__m256 x) {
  // x  = (x7, x6, x5, x4, x3, x2, x1, x0)

  // t0 = (x6, x5, x4, x7, x2, x1, x0, x3)
  __m256 t0 = _mm256_permute_ps(x, _MM_SHUFFLE(2, 1, 0, 3));
  // t1 = ( 0,  0,  0,  0, x6, x5, x4, x7)
  __m256 t1 = _mm256_permute2f128_ps(t0, t0, 0x81);

  //      ( -,  -,  -,  -,  -,  -,  -, x7)
  //      ( 0,  0,  0,  0,  0,  0,  0,  -)
  // y  = ( 0,  0,  0,  0,  0,  0,  0, x7)
  __m256 y = _mm256_blend_ps(t1, _mm256_setzero_ps(), 0xFE /* 0b11111110 */);
  return y;
}
