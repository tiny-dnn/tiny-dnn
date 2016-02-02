/*
    Copyright (c) 2013, Taiga Nomi
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
#include "util.h"
#include "product.h"

#define cnn_restrict __restrict

//haswell
//#define FMADD(a, b, c) _mm256_fmadd_ps(a, b, c)

// ivy/sandy
#define FMADD(a, b, c) _mm256_add_ps(_mm256_mul_ps(a,b),c)


namespace tiny_cnn {


    template <typename T>
    inline void conv2d_nostride_5x5_opt(const T * cnn_restrict src,
        const index3d<layer_size_t>& src_shape,
        const T * cnn_restrict w,
        const index3d<layer_size_t>& w_shape,
        T * cnn_restrict dst,
        const index3d<layer_size_t>& dst_shape)
    {
    }


    // optimized conv for 5x5, no-stride and 8x out-channels
    template <>
    inline void conv2d_nostride_5x5_opt<float>(
        const float * cnn_restrict src,
        const index3d<layer_size_t>& src_shape,
        const float * cnn_restrict weight,
        const index3d<layer_size_t>& w_shape,
        float * cnn_restrict dst,
        const index3d<layer_size_t>& dst_shape)
    {
#define VEC_WIDTH 8
        assert(w_shape.width_ == 5 && w_shape.height_ == 5);
        assert(dst_shape.depth_ % VEC_WIDTH == 0);
        std::fill(dst, dst + dst_shape.size(), 0.0f);

        float *wtmp = (float*)malloc(sizeof(float) * w_shape.size());
        for (int inc = 0; inc < src_shape.depth_; inc++) {
            for (int outc = 0; outc < dst_shape.depth_; outc++) {
                int oi0 = outc % VEC_WIDTH;
                int oi1 = (outc / VEC_WIDTH) * VEC_WIDTH;
                float *wdst = wtmp + (inc * dst_shape.depth_ + oi1) * 25 + oi0;

                for (int i = 0; i < 25; i++)
                    wdst[i * VEC_WIDTH] = weight[(outc * src_shape.depth_ + inc) * 25 + i];

            }
        }


        for (int y = 0; y < dst_shape.height_; y++) {
            for (int x = 0; x < dst_shape.width_; x++) {
                float_t * cnn_restrict d = dst + (y * dst_shape.width_ + x) * dst_shape.depth_;

                // conv-kernel
                for (int inc = 0; inc < src_shape.depth_; inc++) {
                    const float * cnn_restrict i0 = src + src_shape.get_index(x, y + 0, inc);
                    const float * cnn_restrict i1 = src + src_shape.get_index(x, y + 1, inc);
                    const float * cnn_restrict i2 = src + src_shape.get_index(x, y + 2, inc);
                    const float * cnn_restrict i3 = src + src_shape.get_index(x, y + 3, inc);
                    const float * cnn_restrict i4 = src + src_shape.get_index(x, y + 4, inc);

#define INIT_SRC(idx)\
                    __m256 i##idx##0 = _mm256_broadcast_ss(&i##idx[0]);\
                    __m256 i##idx##1 = _mm256_broadcast_ss(&i##idx[1]);\
                    __m256 i##idx##2 = _mm256_broadcast_ss(&i##idx[2]);\
                    __m256 i##idx##3 = _mm256_broadcast_ss(&i##idx[3]);\
                    __m256 i##idx##4 = _mm256_broadcast_ss(&i##idx[4])

                    INIT_SRC(0);
                    INIT_SRC(1);
                    INIT_SRC(2);
                    INIT_SRC(3);
                    INIT_SRC(4);
#undef INIT_SRC

                    float *w = wtmp + inc * dst_shape.depth_ * 25;

                    for (unsigned int outc = 0; outc < dst_shape.depth_; outc += VEC_WIDTH) {
                        __m256 v = _mm256_setzero_ps();

                        v = FMADD(_mm256_loadu_ps(&w[0 * VEC_WIDTH]), i00, v);
                        v = FMADD(_mm256_loadu_ps(&w[1 * VEC_WIDTH]), i01, v);
                        v = FMADD(_mm256_loadu_ps(&w[2 * VEC_WIDTH]), i02, v);
                        v = FMADD(_mm256_loadu_ps(&w[3 * VEC_WIDTH]), i03, v);
                        v = FMADD(_mm256_loadu_ps(&w[4 * VEC_WIDTH]), i04, v);

                        v = FMADD(_mm256_loadu_ps(&w[5 * VEC_WIDTH]), i10, v);
                        v = FMADD(_mm256_loadu_ps(&w[6 * VEC_WIDTH]), i11, v);
                        v = FMADD(_mm256_loadu_ps(&w[7 * VEC_WIDTH]), i12, v);
                        v = FMADD(_mm256_loadu_ps(&w[8 * VEC_WIDTH]), i13, v);
                        v = FMADD(_mm256_loadu_ps(&w[9 * VEC_WIDTH]), i14, v);

                        v = FMADD(_mm256_loadu_ps(&w[10 * VEC_WIDTH]), i20, v);
                        v = FMADD(_mm256_loadu_ps(&w[11 * VEC_WIDTH]), i21, v);
                        v = FMADD(_mm256_loadu_ps(&w[12 * VEC_WIDTH]), i22, v);
                        v = FMADD(_mm256_loadu_ps(&w[13 * VEC_WIDTH]), i23, v);
                        v = FMADD(_mm256_loadu_ps(&w[14 * VEC_WIDTH]), i24, v);

                        v = FMADD(_mm256_loadu_ps(&w[15 * VEC_WIDTH]), i30, v);
                        v = FMADD(_mm256_loadu_ps(&w[16 * VEC_WIDTH]), i31, v);
                        v = FMADD(_mm256_loadu_ps(&w[17 * VEC_WIDTH]), i32, v);
                        v = FMADD(_mm256_loadu_ps(&w[18 * VEC_WIDTH]), i33, v);
                        v = FMADD(_mm256_loadu_ps(&w[19 * VEC_WIDTH]), i34, v);

                        v = FMADD(_mm256_loadu_ps(&w[20 * VEC_WIDTH]), i40, v);
                        v = FMADD(_mm256_loadu_ps(&w[21 * VEC_WIDTH]), i41, v);
                        v = FMADD(_mm256_loadu_ps(&w[22 * VEC_WIDTH]), i42, v);
                        v = FMADD(_mm256_loadu_ps(&w[23 * VEC_WIDTH]), i43, v);
                        v = FMADD(_mm256_loadu_ps(&w[24 * VEC_WIDTH]), i44, v);

                        w += 25 * VEC_WIDTH;
                        __m256 prev = _mm256_loadu_ps(&d[outc]);
                        _mm256_storeu_ps(&d[outc], _mm256_add_ps(prev, v));
                    }

                    /*for (int outc = 0; outc < outd; outc++) {
                        const float * cnn_restrict w = weight + w_shape.get_index(0, 0, outc * src_shape.depth_ + inc);
                        float v = 0;

                        v += w[0] * i0[0];
                        v += w[1] * i0[1];
                        v += w[2] * i0[2];
                        v += w[3] * i0[3];
                        v += w[4] * i0[4];

                        v += w[5] * i1[0];
                        v += w[6] * i1[1];
                        v += w[7] * i1[2];
                        v += w[8] * i1[3];
                        v += w[9] * i1[4];

                        v += w[10] * i2[0];
                        v += w[11] * i2[1];
                        v += w[12] * i2[2];
                        v += w[13] * i2[3];
                        v += w[14] * i2[4];

                        v += w[15] * i3[0];
                        v += w[16] * i3[1];
                        v += w[17] * i3[2];
                        v += w[18] * i3[3];
                        v += w[19] * i3[4];

                        v += w[20] * i4[0];
                        v += w[21] * i4[1];
                        v += w[22] * i4[2];
                        v += w[23] * i4[3];
                        v += w[24] * i4[4];

                        *d += v;
                        d += dst_shape.width_ * dst_shape.height_;
                    }
                    */
                }

                for (int outc = 0; outc < dst_shape.depth_; outc++) {
                    dst[dst_shape.get_index(x, y, outc)] = d[outc];
                }

            }
        }
        free(wtmp);
    }

    inline void conv2d_fallback(const float_t *src,
        const index3d<layer_size_t>& src_shape,
        const float_t *w,
        const index3d<layer_size_t>& w_shape,
        float_t *dst,
        const index3d<layer_size_t>& dst_shape,
        int w_interval,
        int h_interval)
    {
        assert(w_shape.depth_ == src_shape.depth_ * dst_shape.depth_);

        std::fill(dst, dst + dst_shape.size(), float_t(0));

        for (int outc = 0; outc < dst_shape.depth_; outc++) {
            for (int inc = 0; inc < src_shape.depth_; inc++) {
                const float_t* kernel = w + w_shape.get_index(0, 0, outc * src_shape.depth_ + inc);

                for (int y = 0; y < dst_shape.height_; y++) {
                    for (int x = 0; x < dst_shape.width_; x++) {
                        float_t sum = float_t(0);

                        for (int ky = 0; ky < w_shape.height_; ky++)
                            for (int kx = 0; kx < w_shape.width_; kx++)
                                sum += src[src_shape.get_index(x + kx, y + ky, inc)] * kernel[ky * w_shape.width_ + kx];

                        dst[dst_shape.get_index(x, y, outc)] += sum;
                    }
                }
            }
        }
    }

    inline void conv2d(const float_t *src,
        const index3d<layer_size_t>& src_shape,
        const float_t *w,
        const index3d<layer_size_t>& w_shape,
        float_t *dst,
        const index3d<layer_size_t>& dst_shape,
        int w_interval,
        int h_interval)
    {
        if (w_interval == 1 && h_interval == 1 && w_shape.width_ == 5 && w_shape.height_ == 5) {
            // nostride
            conv2d_nostride_5x5_opt<float_t>(src, src_shape, w, w_shape, dst, dst_shape);
            return;
        }


        conv2d_fallback(src, src_shape, w, w_shape, dst, dst_shape, w_interval, h_interval);
    }

} // namespace tiny_cnn