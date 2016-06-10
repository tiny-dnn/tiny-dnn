/*
	This original source code is from http://www.chokkan.org/blog/archives/352
	this modified version requires Xbyak(http://homepage1.nifty.com/herumi/soft/xbyak_e.html)

	g++ -O3 -fomit-frame-pointer -march=core2 -msse4 -fno-operator-names fastexp.cpp -mfpmath=sse fastexp.cpp

1000000 points by NormalRandomGenerator r(0, 1);

iCore i7-2600 3.4GHz + 64-bit Linux + gcc 4.4.5
std::exp     73.850clk, a=171828182.101727
fmath::expd  12.772clk, a=171828182.101727
libc                    67.165039       0.000000e+00    0.000000e+00
Cephes                  27.437542       3.131249e-16    5.754252e-17
Taylor 11th             36.972780       8.792464e-15    1.372958e-15
Taylor 13th             43.597219       2.217901e-16    4.969125e-17
Remez 11th [-0.5,+0.5]  36.953477       4.710267e-16    1.949762e-16
Remez 13th [-0.5,+0.5]  43.704560       2.217901e-16    4.969263e-17
Remez 11th [0,log2]     39.731408       2.571683e-16    6.146323e-17
Remez 13th [0,log2]     46.012650       2.571683e-16    6.101245e-17
Remez 9th [0,log2] SSE  16.290083       1.909579e-14    9.909547e-15
Remez 11th [0,log2]SSE  19.345232       2.571683e-16    6.146323e-17
fmath_expd              11.701059       6.575812e-16    1.117645e-16
fmath_expd_v            8.271875        6.575812e-16    1.117645e-16

iCore i7-2600 3.4GHz + 64-bit Linux + gcc 4.4.5 with -m32 option(32bit)
std::exp     139.405clk, a=171828182.101727
fmath::expd  29.664clk, a=171828182.101727
libc                    133.931580      0.000000e+00    0.000000e+00
Cephes                  28.733528       3.131249e-16    5.750708e-17
Taylor 11th             36.802368       8.792464e-15    1.372957e-15
Taylor 13th             44.288316       2.217901e-16    4.963490e-17
Remez 11th [-0.5,+0.5]  36.685996       4.710267e-16    1.949806e-16
Remez 13th [-0.5,+0.5]  42.937285       2.217901e-16    4.963532e-17
Remez 11th [0,log2]     39.093984       2.571683e-16    6.142713e-17
Remez 13th [0,log2]     45.565401       2.571683e-16    6.097211e-17
Remez 9th [0,log2] SSE  16.653778       1.909579e-14    9.909552e-15
Remez 11th [0,log2]SSE  19.433543       2.571683e-16    6.142713e-17
fmath_expd              29.450048       6.575812e-16    1.117604e-16
fmath_expd_v            7.849043        6.575812e-16    1.117604e-16

iCore i7-2600 3.4GHz + 64-bit Window7 + VC10(64bit)
std::exp     35.060clk, a=171828182.101727
fmath::expd  12.842clk, a=171828182.101727
libc                    35.453626       0.000000e+000   0.000000e+000
Cephes                  33.281304       3.131249e-016   5.741443e-017
Taylor 11th             42.400955       8.792464e-015   1.372950e-015
Taylor 13th             46.394876       2.219280e-016   4.945992e-017
Remez 11th [-0.5,+0.5]  40.276330       4.710267e-016   1.949877e-016
Remez 13th [-0.5,+0.5]  48.644170       2.219280e-016   4.946602e-017
Remez 11th [0,log2]     45.130051       2.571683e-016   6.130098e-017
Remez 13th [0,log2]     51.398456       2.571683e-016   6.084076e-017
Remez 9th [0,log2] SSE  15.344924       1.909579e-014   9.909560e-015
Remez 11th [0,log2]SSE  20.912735       2.571683e-016   6.130098e-017
fmath_expd              14.891861       6.575812e-016   1.117573e-016
fmath_expd_v            7.197682        6.575812e-016   1.117573e-016

iCore i7-2600 3.4GHz + 64-bit Window7 + VC10(32bit)
std::exp     36.945clk, a=171828182.101727
fmath::expd  33.892clk, a=171828182.101727
libc                    23.551640       0.000000e+000   0.000000e+000
Cephes                  36.066613       3.131249e-016   5.748241e-017
Taylor 11th             45.095088       8.792464e-015   1.372974e-015
Taylor 13th             50.380336       2.217703e-016   4.954496e-017
Remez 11th [-0.5,+0.5]  44.772563       4.710267e-016   1.949709e-016
Remez 13th [-0.5,+0.5]  50.710412       2.217703e-016   4.955319e-017
Remez 11th [0,log2]     47.087139       2.624391e-016   6.130986e-017
Remez 13th [0,log2]     52.870111       2.571683e-016   6.088609e-017
Remez 9th [0,log2] SSE  15.803188       1.909579e-014   9.909544e-015
Remez 11th [0,log2]SSE  19.122887       2.624391e-016   6.130986e-017
fmath_expd              27.823901       6.575812e-016   1.117501e-016
fmath_expd_v            8.447081        6.575812e-016   1.117501e-016

iCore i7-2600 3.4GHz + 64-bit Linux + icc 12.0.5(64bit)
std::exp     31.204clk, a=171828182.101727
fmath::expd  11.521clk, a=171828182.101727
libc                    11.831210       0.000000e+00    0.000000e+00
Cephes                  30.851041       6.889636e-16    1.486862e-16
Taylor 11th             34.734583       8.949452e-15    1.383519e-15
Taylor 13th             42.505048       6.355151e-16    1.221867e-16
Remez 11th [-0.5,+0.5]  33.863438       7.965272e-16    2.243559e-16
Remez 13th [-0.5,+0.5]  41.536812       6.355151e-16    1.221242e-16
Remez 11th [0,log2]     38.135168       6.731449e-16    1.259717e-16
Remez 13th [0,log2]     46.723956       6.731449e-16    1.258796e-16
Remez 9th [0,log2] SSE  15.633974       1.909579e-14    9.920769e-15
Remez 11th [0,log2]SSE  18.926368       4.759811e-16    1.133419e-16
fmath_expd              13.430363       8.280873e-16    1.527423e-16
fmath_expd_v            8.235262        7.788500e-16    1.404841e-16

Xeon X5650 2.67GHz + 64-bit Linux + gcc 4.6.0
std::exp     85.425clk, a=171828182.101727
fmath::expd  17.399clk, a=171828182.101727
libc                    67.607064       0.000000e+00    0.000000e+00
Cephes                  33.197321       3.131249e-16    5.754252e-17
Taylor 11th             45.670196       8.792464e-15    1.372958e-15
Taylor 13th             55.180563       2.217901e-16    4.969125e-17
Remez 11th [-0.5,+0.5]  45.634080       4.710267e-16    1.949762e-16
Remez 13th [-0.5,+0.5]  55.379164       2.217901e-16    4.969263e-17
Remez 11th [0,log2]     46.498202       2.571683e-16    6.146323e-17
Remez 13th [0,log2]     56.051165       2.571683e-16    6.101245e-17
Remez 9th [0,log2] SSE  16.185618       1.909579e-14    9.909547e-15
Remez 11th [0,log2]SSE  19.048939       2.571683e-16    6.146323e-17
fmath_expd              14.384895       6.575812e-16    1.117645e-16
fmath_expd_v            9.430332        6.575812e-16    1.117645e-16

Core2Duo 1.8GHz + 64-bit Windows 7 + VC10 (64bit)
libc                    42.743817       0.000000e+000   0.000000e+000
Cephes                  65.548440       3.131249e-016   5.741443e-017
Taylor 11th             76.656996       8.792464e-015   1.372950e-015
Taylor 13th             88.454133       2.219280e-016   4.945992e-017
Remez 11th [-0.5,+0.5]  76.714515       4.710267e-016   1.949877e-016
Remez 13th [-0.5,+0.5]  88.299981       2.219280e-016   4.946602e-017
Remez 11th [0,log2]     86.686488       2.571683e-016   6.130098e-017
Remez 13th [0,log2]     104.098941      2.571683e-016   6.084076e-017
Remez 9th [0,log2] SSE  24.056937       1.909579e-014   9.909560e-015
Remez 11th [0,log2]SSE  27.949167       2.571683e-016   6.130098e-017
fmath_expd              16.556031       6.575812e-016   1.117573e-016
fmath_expd_v            10.149759       6.575812e-016   1.117573e-016

Core2Duo 1.8GHz + 64-bit Windows 7 + VC10 (32bit)
libc                    46.032336       0.000000e+000   0.000000e+000
Cephes                  73.006776       3.131249e-016   5.741443e-017
Taylor 11th             81.204903       8.792464e-015   1.372950e-015
Taylor 13th             88.404687       2.219280e-016   4.945992e-017
Remez 11th [-0.5,+0.5]  77.354901       4.710267e-016   1.949877e-016
Remez 13th [-0.5,+0.5]  88.833690       2.219280e-016   4.946602e-017
Remez 11th [0,log2]     85.900140       2.571683e-016   6.130098e-017
Remez 13th [0,log2]     97.869240       2.571683e-016   6.084076e-017
Remez 9th [0,log2] SSE  21.664476       1.909579e-014   9.909560e-015
Remez 11th [0,log2]SSE  27.176220       2.571683e-016   6.130098e-017
fmath_expd              18.391599       6.575812e-016   1.117573e-016
fmath_expd_v            11.118978       6.575812e-016   1.117573e-016
*/
/*
 *      Fast exp(x) computation (with SSE2 optimizations).
 *
 * Copyright (c) 2010, Naoaki Okazaki
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the names of the authors nor the names of its contributors
 *       may be used to endorse or promote products derived from this
 *       software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*

To compile this code using gcc:
gcc -o fastexp -O3 -fomit-frame-pointer -msse2 -mfpmath=sse -ffast-math -lm fastexp.c

To compile this code using Microsoft Visual C++ 2008 (or later):
Simply create a console project, and add this file to the project.

*/

#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <emmintrin.h>

#include "fmath.hpp"
#define XBYAK_NO_OP_NAMES
#include "xbyak/xbyak_util.h"
/*
    Useful macro definitions for memory alignment:
        http://homepage1.nifty.com/herumi/prog/gcc-and-vc.html#MIE_ALIGN
 */

#ifdef _MSC_VER
#define MIE_ALIGN(x) __declspec(align(x))
#else
#define MIE_ALIGN(x) __attribute__((aligned(x)))
#endif

#ifdef _MSC_VER
    #include <malloc.h>
#else
    #include <stdlib.h>
    static inline void *_aligned_malloc(size_t size, size_t alignment)
    {
        void *p;
        int ret = posix_memalign(&p, alignment, size);
        return (ret == 0) ? p : 0;
    }
#endif

#define CONST_128D(var, val) \
    MIE_ALIGN(16) static const double var[2] = {(val), (val)}
#define CONST_128I(var, v1, v2, v3, v4) \
    MIE_ALIGN(16) static const int var[4] = {(v1), (v2), (v3), (v4)}

typedef struct {
    const char *name;
    void (*func)(double *values, size_t n);
    double error_peak;
    double error_rms;
    long long elapsed_time;
    double *values;
} performance_t;

typedef union {
    double d;
    unsigned short s[4];
} ieee754;

static const double MAXLOG =  7.08396418532264106224E2;     /* log 2**1022 */
static const double MINLOG = -7.08396418532264106224E2;     /* log 2**-1022 */
static const double LOG2E  =  1.4426950408889634073599;     /* 1/log(2) */
//static const double INFINITY = 1.79769313486231570815E308;
static const double C1 = 6.93145751953125E-1;
static const double C2 = 1.42860682030941723212E-6;

void remez9_0_log2_sse(double *values, size_t num)
{
    size_t i;
    CONST_128D(one, 1.);
    CONST_128D(log2e, 1.4426950408889634073599);
    CONST_128D(maxlog, 7.09782712893383996843e2);   // log(2**1024)
    CONST_128D(minlog, -7.08396418532264106224e2);  // log(2**-1022)
    CONST_128D(c1, 6.93145751953125E-1);
    CONST_128D(c2, 1.42860682030941723212E-6);
    CONST_128D(w9, 3.9099787920346160288874633639268318097077213911751e-6);
    CONST_128D(w8, 2.299608440919942766555719515783308016700833740918e-5);
    CONST_128D(w7, 1.99930498409474044486498978862963995247838069436646e-4);
    CONST_128D(w6, 1.38812674551586429265054343505879910146775323730237e-3);
    CONST_128D(w5, 8.3335688409829575034112982839739473866857586300664e-3);
    CONST_128D(w4, 4.1666622504201078708502686068113075402683415962893e-2);
    CONST_128D(w3, 0.166666671414320541875332123507829990378055646330574);
    CONST_128D(w2, 0.49999999974109940909767965915362308135415179642286);
    CONST_128D(w1, 1.0000000000054730504284163017295863259125942049362);
    CONST_128D(w0, 0.99999999999998091336479463057053516986466888462081);
    const __m128i offset = _mm_setr_epi32(1023, 1023, 0, 0);

    for (i = 0;i < num;i += 4) {
        __m128i k1, k2;
        __m128d p1, p2;
        __m128d a1, a2;
        __m128d xmm0, xmm1;
        __m128d x1, x2;

        /* Load four double values. */
        xmm0 = _mm_load_pd(maxlog);
        xmm1 = _mm_load_pd(minlog);
		x1 = _mm_load_pd(values+i);
		x2 = _mm_load_pd(values+i+2);
        x1 = _mm_min_pd(x1, xmm0);
        x2 = _mm_min_pd(x2, xmm0);
        x1 = _mm_max_pd(x1, xmm1);
        x2 = _mm_max_pd(x2, xmm1);

        /* a = x / log2; */
        xmm0 = _mm_load_pd(log2e);
        xmm1 = _mm_setzero_pd();
        a1 = _mm_mul_pd(x1, xmm0);
        a2 = _mm_mul_pd(x2, xmm0);

        /* k = (int)floor(a); p = (float)k; */
        p1 = _mm_cmplt_pd(a1, xmm1);
        p2 = _mm_cmplt_pd(a2, xmm1);
        xmm0 = _mm_load_pd(one);
        p1 = _mm_and_pd(p1, xmm0);
        p2 = _mm_and_pd(p2, xmm0);
        a1 = _mm_sub_pd(a1, p1);
        a2 = _mm_sub_pd(a2, p2);
        k1 = _mm_cvttpd_epi32(a1);
        k2 = _mm_cvttpd_epi32(a2);
        p1 = _mm_cvtepi32_pd(k1);
        p2 = _mm_cvtepi32_pd(k2);

        /* x -= p * log2; */
        xmm0 = _mm_load_pd(c1);
        xmm1 = _mm_load_pd(c2);
        a1 = _mm_mul_pd(p1, xmm0);
        a2 = _mm_mul_pd(p2, xmm0);
        x1 = _mm_sub_pd(x1, a1);
        x2 = _mm_sub_pd(x2, a2);
        a1 = _mm_mul_pd(p1, xmm1);
        a2 = _mm_mul_pd(p2, xmm1);
        x1 = _mm_sub_pd(x1, a1);
        x2 = _mm_sub_pd(x2, a2);

        /* Compute e^x using a polynomial approximation. */
        xmm0 = _mm_load_pd(w9);
        xmm1 = _mm_load_pd(w8);
        a1 = _mm_mul_pd(x1, xmm0);
        a2 = _mm_mul_pd(x2, xmm0);
        a1 = _mm_add_pd(a1, xmm1);
        a2 = _mm_add_pd(a2, xmm1);

        xmm0 = _mm_load_pd(w7);
        xmm1 = _mm_load_pd(w6);
        a1 = _mm_mul_pd(a1, x1);
        a2 = _mm_mul_pd(a2, x2);
        a1 = _mm_add_pd(a1, xmm0);
        a2 = _mm_add_pd(a2, xmm0);
        a1 = _mm_mul_pd(a1, x1);
        a2 = _mm_mul_pd(a2, x2);
        a1 = _mm_add_pd(a1, xmm1);
        a2 = _mm_add_pd(a2, xmm1);

        xmm0 = _mm_load_pd(w5);
        xmm1 = _mm_load_pd(w4);
        a1 = _mm_mul_pd(a1, x1);
        a2 = _mm_mul_pd(a2, x2);
        a1 = _mm_add_pd(a1, xmm0);
        a2 = _mm_add_pd(a2, xmm0);
        a1 = _mm_mul_pd(a1, x1);
        a2 = _mm_mul_pd(a2, x2);
        a1 = _mm_add_pd(a1, xmm1);
        a2 = _mm_add_pd(a2, xmm1);

        xmm0 = _mm_load_pd(w3);
        xmm1 = _mm_load_pd(w2);
        a1 = _mm_mul_pd(a1, x1);
        a2 = _mm_mul_pd(a2, x2);
        a1 = _mm_add_pd(a1, xmm0);
        a2 = _mm_add_pd(a2, xmm0);
        a1 = _mm_mul_pd(a1, x1);
        a2 = _mm_mul_pd(a2, x2);
        a1 = _mm_add_pd(a1, xmm1);
        a2 = _mm_add_pd(a2, xmm1);

        xmm0 = _mm_load_pd(w1);
        xmm1 = _mm_load_pd(w0);
        a1 = _mm_mul_pd(a1, x1);
        a2 = _mm_mul_pd(a2, x2);
        a1 = _mm_add_pd(a1, xmm0);
        a2 = _mm_add_pd(a2, xmm0);
        a1 = _mm_mul_pd(a1, x1);
        a2 = _mm_mul_pd(a2, x2);
        a1 = _mm_add_pd(a1, xmm1);
        a2 = _mm_add_pd(a2, xmm1);

        /* p = 2^k; */
        k1 = _mm_add_epi32(k1, offset);
        k2 = _mm_add_epi32(k2, offset);
        k1 = _mm_slli_epi32(k1, 20);
        k2 = _mm_slli_epi32(k2, 20);
        k1 = _mm_shuffle_epi32(k1, _MM_SHUFFLE(1,3,0,2));
        k2 = _mm_shuffle_epi32(k2, _MM_SHUFFLE(1,3,0,2));
        p1 = _mm_castsi128_pd(k1);
        p2 = _mm_castsi128_pd(k2);

        /* a *= 2^k. */
        a1 = _mm_mul_pd(a1, p1);
        a2 = _mm_mul_pd(a2, p2);

        /* Store the results. */
		_mm_store_pd(values+i, a1);
		_mm_store_pd(values+i+2, a2);
    }
}

void remez11_0_log2_sse(double *values, size_t num)
{
    size_t i;
    CONST_128D(one, 1.);
    CONST_128D(log2e, 1.4426950408889634073599);
    CONST_128D(maxlog, 7.09782712893383996843e2);   // log(2**1024)
    CONST_128D(minlog, -7.08396418532264106224e2);  // log(2**-1022)
    CONST_128D(c1, 6.93145751953125E-1);
    CONST_128D(c2, 1.42860682030941723212E-6);
    CONST_128D(w11, 3.5524625185478232665958141148891055719216674475023e-8);
    CONST_128D(w10, 2.5535368519306500343384723775435166753084614063349e-7);
    CONST_128D(w9, 2.77750562801295315877005242757916081614772210463065e-6);
    CONST_128D(w8, 2.47868893393199945541176652007657202642495832996107e-5);
    CONST_128D(w7, 1.98419213985637881240770890090795533564573406893163e-4);
    CONST_128D(w6, 1.3888869684178659239014256260881685824525255547326e-3);
    CONST_128D(w5, 8.3333337052009872221152811550156335074160546333973e-3);
    CONST_128D(w4, 4.1666666621080810610346717440523105184720007971655e-2);
    CONST_128D(w3, 0.166666666669960803484477734308515404418108830469798);
    CONST_128D(w2, 0.499999999999877094481580370323249951329122224389189);
    CONST_128D(w1, 1.0000000000000017952745258419615282194236357388884);
    CONST_128D(w0, 0.99999999999999999566016490920259318691496540598896);
    const __m128i offset = _mm_setr_epi32(1023, 1023, 0, 0);

    for (i = 0;i < num;i += 4) {
        __m128i k1, k2;
        __m128d p1, p2;
        __m128d a1, a2;
        __m128d xmm0, xmm1;
        __m128d x1, x2;

        /* Load four double values. */
        xmm0 = _mm_load_pd(maxlog);
        xmm1 = _mm_load_pd(minlog);
		x1 = _mm_load_pd(values+i);
		x2 = _mm_load_pd(values+i+2);
        x1 = _mm_min_pd(x1, xmm0);
        x2 = _mm_min_pd(x2, xmm0);
        x1 = _mm_max_pd(x1, xmm1);
        x2 = _mm_max_pd(x2, xmm1);

        /* a = x / log2; */
        xmm0 = _mm_load_pd(log2e);
        xmm1 = _mm_setzero_pd();
        a1 = _mm_mul_pd(x1, xmm0);
        a2 = _mm_mul_pd(x2, xmm0);

        /* k = (int)floor(a); p = (float)k; */
        p1 = _mm_cmplt_pd(a1, xmm1);
        p2 = _mm_cmplt_pd(a2, xmm1);
        xmm0 = _mm_load_pd(one);
        p1 = _mm_and_pd(p1, xmm0);
        p2 = _mm_and_pd(p2, xmm0);
        a1 = _mm_sub_pd(a1, p1);
        a2 = _mm_sub_pd(a2, p2);
        k1 = _mm_cvttpd_epi32(a1);
        k2 = _mm_cvttpd_epi32(a2);
        p1 = _mm_cvtepi32_pd(k1);
        p2 = _mm_cvtepi32_pd(k2);

        /* x -= p * log2; */
        xmm0 = _mm_load_pd(c1);
        xmm1 = _mm_load_pd(c2);
        a1 = _mm_mul_pd(p1, xmm0);
        a2 = _mm_mul_pd(p2, xmm0);
        x1 = _mm_sub_pd(x1, a1);
        x2 = _mm_sub_pd(x2, a2);
        a1 = _mm_mul_pd(p1, xmm1);
        a2 = _mm_mul_pd(p2, xmm1);
        x1 = _mm_sub_pd(x1, a1);
        x2 = _mm_sub_pd(x2, a2);

        /* Compute e^x using a polynomial approximation. */
        xmm0 = _mm_load_pd(w11);
        xmm1 = _mm_load_pd(w10);
        a1 = _mm_mul_pd(x1, xmm0);
        a2 = _mm_mul_pd(x2, xmm0);
        a1 = _mm_add_pd(a1, xmm1);
        a2 = _mm_add_pd(a2, xmm1);

        xmm0 = _mm_load_pd(w9);
        xmm1 = _mm_load_pd(w8);
        a1 = _mm_mul_pd(a1, x1);
        a2 = _mm_mul_pd(a2, x2);
        a1 = _mm_add_pd(a1, xmm0);
        a2 = _mm_add_pd(a2, xmm0);
        a1 = _mm_mul_pd(a1, x1);
        a2 = _mm_mul_pd(a2, x2);
        a1 = _mm_add_pd(a1, xmm1);
        a2 = _mm_add_pd(a2, xmm1);

        xmm0 = _mm_load_pd(w7);
        xmm1 = _mm_load_pd(w6);
        a1 = _mm_mul_pd(a1, x1);
        a2 = _mm_mul_pd(a2, x2);
        a1 = _mm_add_pd(a1, xmm0);
        a2 = _mm_add_pd(a2, xmm0);
        a1 = _mm_mul_pd(a1, x1);
        a2 = _mm_mul_pd(a2, x2);
        a1 = _mm_add_pd(a1, xmm1);
        a2 = _mm_add_pd(a2, xmm1);

        xmm0 = _mm_load_pd(w5);
        xmm1 = _mm_load_pd(w4);
        a1 = _mm_mul_pd(a1, x1);
        a2 = _mm_mul_pd(a2, x2);
        a1 = _mm_add_pd(a1, xmm0);
        a2 = _mm_add_pd(a2, xmm0);
        a1 = _mm_mul_pd(a1, x1);
        a2 = _mm_mul_pd(a2, x2);
        a1 = _mm_add_pd(a1, xmm1);
        a2 = _mm_add_pd(a2, xmm1);

        xmm0 = _mm_load_pd(w3);
        xmm1 = _mm_load_pd(w2);
        a1 = _mm_mul_pd(a1, x1);
        a2 = _mm_mul_pd(a2, x2);
        a1 = _mm_add_pd(a1, xmm0);
        a2 = _mm_add_pd(a2, xmm0);
        a1 = _mm_mul_pd(a1, x1);
        a2 = _mm_mul_pd(a2, x2);
        a1 = _mm_add_pd(a1, xmm1);
        a2 = _mm_add_pd(a2, xmm1);

        xmm0 = _mm_load_pd(w1);
        xmm1 = _mm_load_pd(w0);
        a1 = _mm_mul_pd(a1, x1);
        a2 = _mm_mul_pd(a2, x2);
        a1 = _mm_add_pd(a1, xmm0);
        a2 = _mm_add_pd(a2, xmm0);
        a1 = _mm_mul_pd(a1, x1);
        a2 = _mm_mul_pd(a2, x2);
        a1 = _mm_add_pd(a1, xmm1);
        a2 = _mm_add_pd(a2, xmm1);

        /* p = 2^k; */
        k1 = _mm_add_epi32(k1, offset);
        k2 = _mm_add_epi32(k2, offset);
        k1 = _mm_slli_epi32(k1, 20);
        k2 = _mm_slli_epi32(k2, 20);
        k1 = _mm_shuffle_epi32(k1, _MM_SHUFFLE(1,3,0,2));
        k2 = _mm_shuffle_epi32(k2, _MM_SHUFFLE(1,3,0,2));
        p1 = _mm_castsi128_pd(k1);
        p2 = _mm_castsi128_pd(k2);

        /* a *= 2^k. */
        a1 = _mm_mul_pd(a1, p1);
        a2 = _mm_mul_pd(a2, p2);

        /* Store the results. */
		_mm_store_pd(values+i, a1);
		_mm_store_pd(values+i+2, a2);
    }
}

void remez11_0_log2(double *values, size_t num)
{
    size_t i;
    for (i = 0;i < num;++i) {
        int n;
        double a, px, x = values[i];
        ieee754 u;

        /* n = floor(x / log 2) */
        a = LOG2E * x;
        a -= (a < 0);
        n = (int)a;

        /* x -= n * log2 */
        px = (double)n;
        x -= px * C1;
        x -= px * C2;

        /* Compute e^x using a polynomial approximation. */
        a = 3.5524625185478232665958141148891055719216674475023e-8;
        a *= x;
        a += 2.5535368519306500343384723775435166753084614063349e-7;
        a *= x;
        a += 2.77750562801295315877005242757916081614772210463065e-6;
        a *= x;
        a += 2.47868893393199945541176652007657202642495832996107e-5;
        a *= x;
        a += 1.98419213985637881240770890090795533564573406893163e-4;
        a *= x;
        a += 1.3888869684178659239014256260881685824525255547326e-3;
        a *= x;
        a += 8.3333337052009872221152811550156335074160546333973e-3;
        a *= x;
        a += 4.1666666621080810610346717440523105184720007971655e-2;
        a *= x;
        a += 0.166666666669960803484477734308515404418108830469798;
        a *= x;
        a += 0.499999999999877094481580370323249951329122224389189;
        a *= x;
        a += 1.0000000000000017952745258419615282194236357388884;
        a *= x;
        a += 0.99999999999999999566016490920259318691496540598896;

        /* Build 2^n in double. */
        u.d = 0;
        n += 1023;
        u.s[3] = (unsigned short)((n << 4) & 0x7FF0);

        values[i] = a * u.d;
    }
}

void remez13_0_log2(double *values, size_t num)
{
    size_t i;
    for (i = 0;i < num;++i) {
        int n;
        double a, px, x = values[i];
        ieee754 u;

        /* n = floor(x / log 2) */
        a = LOG2E * x;
        a -= (a < 0);
        n = (int)a;

        /* x -= n * log2 */
        px = (double)n;
        x -= px * C1;
        x -= px * C2;

        /* Compute e^x using a polynomial approximation. */
        a = 2.2762932529150460619497906285755631573256951680928e-10;
        a *= x;
        a += 1.93367224471636363463651078554697568501742984768854e-9;
        a *= x;
        a += 2.52543927629810215309605055241333435158089916072836e-8;
        a *= x;
        a += 2.7540144801860636516400552945824228138166510066936e-7;
        a *= x;
        a += 2.75583147053220552447847947870361102182338458956017e-6;
        a *= x;
        a += 2.4801546952196268386551625341270924245448949820144e-5;
        a *= x;
        a += 1.98412709907914555147826749325174275063119441397255e-4;
        a *= x;
        a += 1.38888888661019235625906849288500655817315802824057e-3;
        a *= x;
        a += 8.333333333639598748175600716777066054934952111002e-3;
        a *= x;
        a += 4.16666666666400203177888506088903193769823751590758e-2;
        a *= x;
        a += 0.16666666666666805461760233769080472413024408831279;
        a *= x;
        a += 0.499999999999999962289068852705038969538821467770838;
        a *= x;
        a += 1.0000000000000000004034789178104888993769939051368;
        a *= x;
        a += 0.99999999999999999999928421898456045238677548461656;

        /* Build 2^n in double. */
        u.d = 0;
        n += 1023;
        u.s[3] = (unsigned short)((n << 4) & 0x7FF0);

        values[i] = a * u.d;
    }
}

void vecexp_remez11_05_05(double *values, size_t num)
{
    size_t i;
    for (i = 0;i < num;++i) {
        int n;
        double a, px, x = values[i];
        ieee754 u;

        /* n = round(x / log 2) */
        a = LOG2E * x + 0.5;
        n = (int)a;
        n -= (a < 0);

        /* x -= n * log2 */
        px = (double)n;
        x -= px * C1;
        x -= px * C2;

        /* Compute e^x using a polynomial approximation. */
        a = 2.51929868958558990100711431834993496570205380722036e-8;
        a *= x;
        a += 2.7714302972476470857291821430459163185492029688565e-7;
        a *= x;
        a += 2.7556840833993121572460386718599369519143729132621e-6;
        a *= x;
        a += 2.48011454027599465130934662141475239085795284566129e-5;
        a *= x;
        a += 1.98412706284313544001695569595213030418978110809054e-4;
        a *= x;
        a += 1.38888894619673011407756064437561180652889917616113e-3;
        a *= x;
        a += 8.3333333326960625302778843136952722007056714994369e-3;
        a *= x;
        a += 4.1666666663307925829807454939715320634892173110281e-2;
        a *= x;
        a += 0.16666666666668912253462202236231047014798117425631;
        a *= x;
        a += 0.50000000000007198509320027789925087826877822951327;
        a *= x;
        a += 0.99999999999999976925988096235016345824070346859229;
        a *= x;
        a += 0.9999999999999997500224010478732616185808429624127;

        /* Build 2^n in double. */
        u.d = 0;
        n += 1023;
        u.s[3] = (unsigned short)((n << 4) & 0x7FF0);

        values[i] = a * u.d;
    }
}

void vecexp_remez13_05_05(double *values, size_t num)
{
    size_t i;
    for (i = 0;i < num;++i) {
        int n;
        double a, px, x = values[i];
        ieee754 u;

        /* n = round(x / log 2) */
        a = LOG2E * x + 0.5;
        n = (int)a;
        n -= (a < 0);

        /* x -= n * log2 */
        px = (double)n;
        x -= px * C1;
        x -= px * C2;

        /* Compute e^x using a polynomial approximation. */
        a = 1.61356848991757396195626449324190427883666354504187e-10;
        a *= x;
        a += 2.09773502429720720042832865218235470540902290986604e-9;
        a *= x;
        a += 2.50517997973487340520345973228530569065189960125308e-8;
        a *= x;
        a += 2.7556973177905466877853160305636358948188615327725e-7;
        a *= x;
        a += 2.7557319860870868162733628841564031959051602041504e-6;
        a *= x;
        a += 2.48015878916569323181318984106104816451563207545696e-5;
        a *= x;
        a += 1.98412698405602139306149764767811806786694849547848e-4;
        a *= x;
        a += 1.38888888883724638191056667131613472090590184962952e-3;
        a *= x;
        a += 8.333333333333743243093579329746662890641071879753e-3;
        a *= x;
        a += 4.16666666666688187524214985734081300874557496783939e-2;
        a *= x;
        a += 0.16666666666666665609766712753641211515446088211247;
        a *= x;
        a += 0.49999999999999996637019704330727593814705189927683;
        a *= x;
        a += 1.00000000000000000008007233620462705038544342452684;
        a *= x;
        a += 1.0000000000000000000857966908786376708355989802095;

        /* Build 2^n in double. */
        u.d = 0;
        n += 1023;
        u.s[3] = (unsigned short)((n << 4) & 0x7FF0);

        values[i] = a * u.d;
    }
}

void vecexp_taylor11(double *values, size_t num)
{
    size_t i;
    for (i = 0;i < num;++i) {
        int n;
        double a, px, x = values[i];
        ieee754 u;

        /* n = round(x / log 2) */
        a = LOG2E * x + 0.5;
        n = (int)a;
        n -= (a < 0);

        /* x -= n * log2 */
        px = (double)n;
        x -= px * C1;
        x -= px * C2;

        /* Compute e^x using a polynomial approximation. */
        a = 1. / 39916800.;
        a *= x;
        a += 2.7557319223985890652557319223985890652557319223986e-7;
        a *= x;
        a += 2.75573192239858906525573192239858906525573192239859e-6;
        a *= x;
        a += 2.4801587301587301587301587301587301587301587301587e-5;
        a *= x;
        a += 1.98412698412698412698412698412698412698412698412698e-4;
        a *= x;
        a += 1.38888888888888888888888888888888888888888888888889e-3;
        a *= x;
        a += 8.3333333333333333333333333333333333333333333333333e-3;
        a *= x;
        a += 4.1666666666666666666666666666666666666666666666666e-2;
        a *= x;
        a += 0.166666666666666666666666666666666666666666666666665;
        a *= x;
        a += 0.5;
        a *= x;
        a += 1.0;
        a *= x;
        a += 1.0;

        /* Build 2^n in double. */
        u.d = 0;
        n += 1023;
        u.s[3] = (unsigned short)((n << 4) & 0x7FF0);

        values[i] = a * u.d;
    }
}

void vecexp_taylor13(double *values, size_t num)
{
    size_t i;
    for (i = 0;i < num;++i) {
        int n;
        double a, px, x = values[i];
        ieee754 u;

        /* n = round(x / log 2) */
        a = LOG2E * x + 0.5;
        n = (int)a;
        n -= (a < 0);

        /* x -= n * log2 */
        px = (double)n;
        x -= px * C1;
        x -= px * C2;

        /* Compute e^x using a polynomial approximation. */
        a = 1. / 6227020800LL;
        a *= x;
        a += 2.08767569878680989792100903212014323125434236545349e-9;
        a *= x;
        a += 2.50521083854417187750521083854417187750521083854419e-8;
        a *= x;
        a += 2.7557319223985890652557319223985890652557319223986e-7;
        a *= x;
        a += 2.75573192239858906525573192239858906525573192239859e-6;
        a *= x;
        a += 2.4801587301587301587301587301587301587301587301587e-5;
        a *= x;
        a += 1.98412698412698412698412698412698412698412698412698e-4;
        a *= x;
        a += 1.38888888888888888888888888888888888888888888888889e-3;
        a *= x;
        a += 8.3333333333333333333333333333333333333333333333333e-3;
        a *= x;
        a += 4.1666666666666666666666666666666666666666666666666e-2;
        a *= x;
        a += 0.166666666666666666666666666666666666666666666666665;
        a *= x;
        a += 0.5;
        a *= x;
        a += 1.0;
        a *= x;
        a += 1.0;

        /* Build 2^n in double. */
        u.d = 0;
        n += 1023;
        u.s[3] = (unsigned short)((n << 4) & 0x7FF0);

        values[i] = a * u.d;
    }
}

void vecexp_cephes(double *values, size_t num)
{
    size_t i;
    for (i = 0;i < num;++i) {
        int n;
        double x = values[i];
        double a, xx, px, qx;
        ieee754 u;

        /* n = round(x / log 2) */
        a = LOG2E * x + 0.5;
        n = (int)a;
        n -= (a < 0);

        /* x -= n * log2 */
        px = (double)n;
        x -= px * C1;
        x -= px * C2;
        xx = x * x;

        /* px = x * P(x**2). */
        px = 1.26177193074810590878E-4;
        px *= xx;
        px += 3.02994407707441961300E-2;
        px *= xx;
        px += 9.99999999999999999910E-1;
        px *= x;

        /* Evaluate Q(x**2). */
        qx = 3.00198505138664455042E-6;
        qx *= xx;
        qx += 2.52448340349684104192E-3;
        qx *= xx;
        qx += 2.27265548208155028766E-1;
        qx *= xx;
        qx += 2.00000000000000000009E0;

        /* e**x = 1 + 2x P(x**2)/( Q(x**2) - P(x**2) ) */
        x = px / (qx - px);
        x = 1.0 + 2.0 * x;

        /* Build 2^n in double. */
        u.d = 0;
        n += 1023;
        u.s[3] = (unsigned short)((n << 4) & 0x7FF0);

        values[i] = x * u.d;
    }
}

void vecexp_libc(double *values, size_t n)
{
    size_t i;
    for (i = 0;i < n;++i) {
        values[i] = exp(values[i]);
    }
}

class RandomGenerator {
    unsigned int x_, y_, z_, w_;
public:
    RandomGenerator(int seed = 0)
    {
        init(seed);
    }
    void init(int seed = 0)
    {
        x_ = 123456789 + seed;
        y_ = 362436069;
        z_ = 521288629;
        w_ = 88675123;
    }
    unsigned int get()
    {
        unsigned int t = x_ ^ (x_ << 11);
        x_ = y_; y_ = z_; z_ = w_;
        return w_ = (w_ ^ (w_ >> 19)) ^ (t ^ (t >> 8));
    }
};
/*
    normal random generator
*/
class NormalRandomGenerator {
    RandomGenerator gen_;
    double u_;
    double s_;
public:
    NormalRandomGenerator(double u = 0, double s = 1, int seed = 0)
        : gen_(seed)
        , u_(u)
        , s_(s)
    {
    }
    void init(int seed = 0)
    {
        gen_.init(seed);
    }
    double get()
    {
        double sum = -6;
        for (int i = 0; i < 12; i++) {
            sum += gen_.get() / double(1ULL << 32);
        }
        return sum * s_ + u_;
    }
};

double *read_source(int *num)
{
	const int n = 1000000;
	*num = n;
	NormalRandomGenerator r(0, 1);
	double *values = (double*)malloc(n * sizeof(double));
	for (int i = 0; i < n; i++) {
		values[i] = r.get();
	}
	return values;
}

void measure(performance_t *perf, double *values, size_t n)
{
    size_t i;
    performance_t *p;

    for (p = perf;p->func != NULL;++p) {
        p->values = (double*)_aligned_malloc(sizeof(double) * n, 32);
        for (i = 0;i < n;++i) {
            p->values[i] = values[i];
        }
    }

    for (p = perf;p->func != NULL;++p) {
        Xbyak::util::Clock clk;
        clk.begin();
        p->func(p->values, n);
        clk.end();
        p->elapsed_time = clk.getClock();
    }

    for (p = perf;p->func != NULL;++p) {
        for (i = 0;i < n;++i) {
            double ex = perf[0].values[i];
            double exf = p->values[i];

            double err = fabs(exf - ex) / ex;
            if (p->error_peak < err) {
                p->error_peak = err;
            }
            p->error_rms += (err * err);
        }

        p->error_rms /= n;
        p->error_rms = sqrt(p->error_rms);
    }
}

void fmath_expd(double *values, size_t n)
{
    for (size_t i = 0;i < n; i++) {
        values[i] = fmath::expd(values[i]);
    }
}

void benchmark(const char *str, double f(double))
{
	double a = 0;
	Xbyak::util::Clock clk;
	clk.begin();
	int n = 0;
	for (double x = 0; x < 1; x += 1e-8) {
		a += f(x);
		n++;
	}
	clk.end();
	printf("%s %.3fclk, a=%f\n", str, clk.getClock() / double(n), a);
}

void compare(double x)
{
	double a = exp(x);
	double b = fmath::expd(x);
	double diff = fabs(a - b);
	if (diff > 1e-13 && fabs(a - b) / a > 1e-13) {
		printf("x=%.17g a=%.17g b=%.17g\n", x, a, b);
	}
}

void testLimits()
{
	const int N = 10000;
	for (int i = 0; i < N; i++) {
		double x = 709 + i / double(N);
		compare(x);
	}
	for (int i = 0; i < N; i++) {
		double x = -708 - i / double(N);
		compare(x);
	}
}

void testSlots()
{
	float vals[4] = {0.1f, 20.0f, -20.0f, -40.0f};
	float expvals[4] = {0.0f};
	for (int i = 0; i < 4; i++) {
		__m128 x = _mm_set_ps(vals[(i + 3) % 4], vals[(i + 2) % 4], vals[(i + 1) % 4], vals[(i + 0) % 4]);
		_mm_storeu_ps(expvals, fmath::exp_ps(x));
		for (int j = 0; j < 4; j++) {
			int idx = (i + j) % 4;
			float expect = fmath::exp(vals[idx]);
			if (fabs(expect - expvals[j]) > 1e-13f) {
				printf("%d: expect[%d]=%.17g != shuffled[%d]=%.17g\n", i, idx, expect, j, expvals[j]);
			}
		}
	}
}

int main()
{
	testLimits();
	testSlots();
	benchmark("std::exp    ", ::exp);
	benchmark("fmath::expd ", fmath::expd);

    int n;
    double *values = NULL;
    performance_t *p = NULL;

    performance_t perf[] = {
        {"libc                  ", vecexp_libc, 0., 0., 0, NULL},
        {"Cephes                ", vecexp_cephes, 0., 0., 0, NULL},
        {"Taylor 11th           ", vecexp_taylor11, 0., 0., 0, NULL},
        {"Taylor 13th           ", vecexp_taylor13, 0., 0., 0, NULL},
        {"Remez 11th [-0.5,+0.5]", vecexp_remez11_05_05, 0., 0., 0, NULL},
        {"Remez 13th [-0.5,+0.5]", vecexp_remez13_05_05, 0., 0., 0, NULL},
        {"Remez 11th [0,log2]   ", remez11_0_log2, 0., 0., 0, NULL},
        {"Remez 13th [0,log2]   ", remez13_0_log2, 0., 0., 0, NULL},
        {"Remez 9th [0,log2] SSE", remez9_0_log2_sse, 0., 0., 0, NULL},
        {"Remez 11th [0,log2]SSE", remez11_0_log2_sse, 0., 0., 0, NULL},
        {"fmath_expd            ", fmath_expd, 0., 0., 0, NULL},
        {"fmath_expd_v          ", fmath::expd_v, 0., 0., 0, NULL},
        {NULL, NULL, 0., 0., 0, NULL},
    };

    values = read_source(&n);
    measure(perf, values, n);

    for (p = perf;p->func != NULL;++p) {
        printf(
            "%s\t%f\t%e\t%e\n",
            p->name,
            p->elapsed_time / (double)n,
            p->error_peak,
            p->error_rms
            );
    }
}

