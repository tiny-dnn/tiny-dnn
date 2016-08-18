/*
    Copyright (c) 2016, Taiga Nomi, Edgar Riba
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#ifndef CNN_USE_AVX
#error Advanced Vector Extensions required.
#endif

#ifndef _mm256_set_m128
#define _mm256_set_m128(va, vb) \
        _mm256_insertf128_ps(_mm256_castps128_ps256(vb), va, 1)
#endif

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

// Horizontally add elements of __m256 type argument (sadly, _mm256_hadd_ps isn't good enough)
// http://stackoverflow.com/a/13222410/4699324
// x = ( x7, x6, x5, x4, x3, x2, x1, x0 )
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

template<int n>
struct foobar : std::false_type
{ };


// Byte Shift YMM Register Across 128-bit Lanes
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
    __m256 y = _mm256_blend_ps(t0, t1, 0x33 /* 0b00110011 */ );
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
    __m256 y = _mm256_blend_ps(t0, t1, 0x77 /* 0b01110111 */ );
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
    __m256 y = _mm256_blend_ps(t1, _mm256_setzero_ps(), 0x10 /* 0b00010000 */ );
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
    __m256 y = _mm256_blend_ps(_mm256_setzero_ps(), t1, 0xC0 /* 0b11000000 */ );
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
    __m256 y = _mm256_blend_ps(_mm256_setzero_ps(), t1, 0x80 /* 0b10000000 */ );
    return y;
}

template <int n>
inline __m256 rightShift(__m256 a)
{
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
    __m256 y = _mm256_blend_ps(t0, t1, 0x88 /* 0b10001000 */ );
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
    __m256 y = _mm256_blend_ps(t0, t1, 0xCC /* 0b11001100 */ );
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
    __m256 y = _mm256_blend_ps(t0, t1, 0xEE /* 0b11101110 */ );
    return y;
}

template <>
inline __m256 rightShift<16>(__m256 x)
{
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
    __m256 y = _mm256_blend_ps(t1, _mm256_setzero_ps(), 0xF8 /* 0b11111000 */ );
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
    __m256 y = _mm256_blend_ps(t1, _mm256_setzero_ps(), 0xFC /* 0b11111100 */ );
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
    __m256 y = _mm256_blend_ps(t1, _mm256_setzero_ps(), 0xFE /* 0b11111110 */ );
    return y;
}

