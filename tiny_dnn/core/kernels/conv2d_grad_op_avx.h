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

#include <vector>
#include "tiny_dnn/core/params/conv_params.h"
#include "tiny_dnn/core/kernels/conv2d_op_internal.h"

#ifdef CNN_USE_AVX
#include "tiny_dnn/core/kernels/avx_kernel_common.h"
#endif

namespace tiny_dnn {
namespace kernels {

#ifdef CNN_USE_AVX

// float ver
template <typename Allocator>
void avx_conv2d_5x5_back_kernel_one(const core::conv_params& params,
                                    const std::vector<float, Allocator>& prev_out,
                                    const std::vector<float, Allocator>& W,
                                    std::vector<float, Allocator>&       dW,
                                    std::vector<float, Allocator>&       db,
                                    std::vector<float, Allocator>&       curr_delta,
                                    std::vector<float, Allocator>*       prev_delta) {
    auto& in        = params.in;
    auto& out       = params.out;
    auto& in_padded = params.in_padded;
    auto& tbl       = params.tbl;
    auto  w_stride  = params.w_stride;
    const size_t in_padded_area = in_padded.area();
    float* pdelta_dst_org = &(*prev_delta)[0];
    const size_t  h_stride2 = params.h_stride * in_padded.width_;
    static const __m256i imask = _mm256_setr_epi32(-1, -1, -1, -1, -1, 0, 0, 0);
    static const __m256 mask = _mm256_castsi256_ps(_mm256_setr_epi32(-1, -1, -1, -1, -1, 0, 0, 0));
    // propagate delta to previous layer
    if (w_stride == 1 && out.width_ >= 4) {
        const serial_size_t nblocks = out.width_ / 4;
        for (serial_size_t inc = 0; inc < in.depth_; ++inc, pdelta_dst_org += in_padded_area) {
            for (serial_size_t outc = 0; outc < out.depth_; outc++) {
                if (!tbl.is_connected(outc, inc)) continue;
                const float* pw = &W[25 * (in.depth_ * outc + inc)];
                const float* pdelta_src = &curr_delta[out.get_index(0, 0, outc)];
                float* pdelta_dst = pdelta_dst_org;
                __m256 w0a = _mm256_and_ps(_mm256_loadu_ps(pw+0), mask);
                __m256 w1a = _mm256_and_ps(_mm256_loadu_ps(pw+5), mask);
                __m256 w2a = _mm256_and_ps(_mm256_loadu_ps(pw+10), mask);
                __m256 w3a = _mm256_and_ps(_mm256_loadu_ps(pw+15), mask);
                __m256 w4a = _mm256_and_ps(_mm256_loadu_ps(pw+20), mask);
                __m256 w0b = leftShift<4>(w0a);
                __m256 w1b = leftShift<4>(w1a);
                __m256 w2b = leftShift<4>(w2a);
                __m256 w3b = leftShift<4>(w3a);
                __m256 w4b = leftShift<4>(w4a);
                __m256 w0c = leftShift<8>(w0a);
                __m256 w1c = leftShift<8>(w1a);
                __m256 w2c = leftShift<8>(w2a);
                __m256 w3c = leftShift<8>(w3a);
                __m256 w4c = leftShift<8>(w4a);
                __m256 w0d = leftShift<12>(w0a);
                __m256 w1d = leftShift<12>(w1a);
                __m256 w2d = leftShift<12>(w2a);
                __m256 w3d = leftShift<12>(w3a);
                __m256 w4d = leftShift<12>(w4a);
                for (serial_size_t y = 0; y < out.height_; y++) {
                    const float* pdelta_src2 = pdelta_src;
                    float* delta_dst0 = pdelta_dst;
                    float* delta_dst1 = &pdelta_dst[in_padded.width_ * 1];
                    float* delta_dst2 = &pdelta_dst[in_padded.width_ * 2];
                    float* delta_dst3 = &pdelta_dst[in_padded.width_ * 3];
                    float* delta_dst4 = &pdelta_dst[in_padded.width_ * 4];
                    for (serial_size_t n = 0; n < nblocks; ++n) {
                        __m256 delta_src = _mm256_broadcast_ps((const __m128*)pdelta_src2);
                        __m256 dst0 = _mm256_loadu_ps(delta_dst0 + 4 * n);
                        __m256 dst1 = _mm256_loadu_ps(delta_dst1 + 4 * n);
                        __m256 dst2 = _mm256_loadu_ps(delta_dst2 + 4 * n);
                        __m256 dst3 = _mm256_loadu_ps(delta_dst3 + 4 * n);
                        __m256 dst4 = _mm256_loadu_ps(delta_dst4 + 4 * n);
                        __m256 delta_src0 = _mm256_permute_ps(delta_src, _MM_SHUFFLE(0, 0, 0, 0));
                        __m256 delta_src1 = _mm256_permute_ps(delta_src, _MM_SHUFFLE(1, 1, 1, 1));
                        __m256 delta_src2 = _mm256_permute_ps(delta_src, _MM_SHUFFLE(2, 2, 2, 2));
                        __m256 delta_src3 = _mm256_permute_ps(delta_src, _MM_SHUFFLE(3, 3, 3, 3));
                        dst0 = madd256_ps(w0a, delta_src0, dst0);
                        dst1 = madd256_ps(w1a, delta_src0, dst1);
                        dst2 = madd256_ps(w2a, delta_src0, dst2);
                        dst3 = madd256_ps(w3a, delta_src0, dst3);
                        dst4 = madd256_ps(w4a, delta_src0, dst4);
                        dst0 = madd256_ps(w0b, delta_src1, dst0);
                        dst1 = madd256_ps(w1b, delta_src1, dst1);
                        dst2 = madd256_ps(w2b, delta_src1, dst2);
                        dst3 = madd256_ps(w3b, delta_src1, dst3);
                        dst4 = madd256_ps(w4b, delta_src1, dst4);
                        dst0 = madd256_ps(w0c, delta_src2, dst0);
                        dst1 = madd256_ps(w1c, delta_src2, dst1);
                        dst2 = madd256_ps(w2c, delta_src2, dst2);
                        dst3 = madd256_ps(w3c, delta_src2, dst3);
                        dst4 = madd256_ps(w4c, delta_src2, dst4);
                        dst0 = madd256_ps(w0d, delta_src3, dst0);
                        _mm256_storeu_ps(delta_dst0 + 4 * n, dst0);
                        dst1 = madd256_ps(w1d, delta_src3, dst1);
                        _mm256_storeu_ps(delta_dst1 + 4 * n, dst1);
                        dst2 = madd256_ps(w2d, delta_src3, dst2);
                        _mm256_storeu_ps(delta_dst2 + 4 * n, dst2);
                        dst3 = madd256_ps(w3d, delta_src3, dst3);
                        _mm256_storeu_ps(delta_dst3 + 4 * n, dst3);
                        dst4 = madd256_ps(w4d, delta_src3, dst4);
                        _mm256_storeu_ps(delta_dst4 + 4 * n, dst4);
                        pdelta_src2 += 4;
                    }
                    for (serial_size_t x = nblocks * 4; x < out.width_; x++) {
                        __m256 delta_src = _mm256_broadcast_ss(pdelta_src + x);
                        __m256 dst0 = _mm256_loadu_ps(delta_dst0 + x);
                        __m256 dst1 = _mm256_loadu_ps(delta_dst1 + x);
                        __m256 dst2 = _mm256_loadu_ps(delta_dst2 + x);
                        __m256 dst3 = _mm256_loadu_ps(delta_dst3 + x);
                        __m256 dst4 = _mm256_loadu_ps(delta_dst4 + x);
                        dst0 = madd256_ps(w0a, delta_src, dst0);
                        dst1 = madd256_ps(w1a, delta_src, dst1);
                        dst2 = madd256_ps(w2a, delta_src, dst2);
                        dst3 = madd256_ps(w3a, delta_src, dst3);
                        dst4 = madd256_ps(w4a, delta_src, dst4);
                        _mm256_storeu_ps(delta_dst0 + x, dst0);
                        _mm256_storeu_ps(delta_dst1 + x, dst1);
                        _mm256_storeu_ps(delta_dst2 + x, dst2);
                        _mm256_storeu_ps(delta_dst3 + x, dst3);
                        _mm256_storeu_ps(delta_dst4 + x, dst4);
                    }
                    pdelta_src += out.width_;
                    pdelta_dst += h_stride2;
                }
            }
        }
    } else if (out.height_ == 1 && out.width_ == 1) {
        for (serial_size_t inc = 0; inc < in.depth_; ++inc, pdelta_dst_org += in_padded_area) {
            float* delta_dst0 = pdelta_dst_org;
            float* delta_dst1 = &pdelta_dst_org[in_padded.width_ * 1];
            float* delta_dst2 = &pdelta_dst_org[in_padded.width_ * 2];
            float* delta_dst3 = &pdelta_dst_org[in_padded.width_ * 3];
            float* delta_dst4 = &pdelta_dst_org[in_padded.width_ * 4];
            __m256 dst0 = _mm256_loadu_ps(delta_dst0);
            __m256 dst1 = _mm256_loadu_ps(delta_dst1);
            __m256 dst2 = _mm256_loadu_ps(delta_dst2);
            __m256 dst3 = _mm256_loadu_ps(delta_dst3);
            __m256 dst4 = _mm256_maskload_ps(delta_dst4, imask);

            // *FROM
            // ---0 0000
            // ---1 1111
            // ---2 2222
            // ---3 3333
            // ---4 4444
            //
            // *TO
            // 1110 0000
            // 3222 2211
            // 4444 3333
            // ---- ---4
            __m256 sum0 = _mm256_blend_ps(
                dst0,
                leftShift<20>(dst1),
                0xE0 /* 0b11100000 */
            );
            __m256 sum1 = _mm256_blend_ps(
                leftShift<28>(dst3),
                _mm256_blend_ps(leftShift<8>(dst2), rightShift<12>(dst1), 0x03 /* 0b00000011 */),
                0x7F /* 0b01111111 */
            );
            __m256 sum2 = _mm256_blend_ps(
                leftShift<16>(dst4),
                rightShift<4>(dst3),
                0x0F /* 0b00001111 */
            );
            __m128 sum3 = _mm256_extractf128_ps(dst4, 1);

            size_t widx = 25 * inc;
            size_t wstep = 25 * in.depth_;

            if (tbl.is_empty()) {
                for (serial_size_t outc = 0; outc < out.depth_; outc++, widx+=wstep) {
                    __m256 delta_src = _mm256_broadcast_ss(&curr_delta[outc]);
                    const float* pw = (const float*)&W[widx];
                    __m256 w0 = _mm256_loadu_ps(pw+0);
                    __m256 w1 = _mm256_loadu_ps(pw + 8);
                    __m256 w2 = _mm256_loadu_ps(pw + 16);
                    __m128 w3 = _mm_load_ss(pw + 24);
                    sum0 = madd256_ps(w0, delta_src, sum0);
                    sum1 = madd256_ps(w1, delta_src, sum1);
                    sum2 = madd256_ps(w2, delta_src, sum2);
                    sum3 = madd128_ss(w3, _mm256_castps256_ps128(delta_src), sum3);
                }
            }
            else {
                for (serial_size_t outc = 0; outc < out.depth_; outc++, widx += wstep) {
                    if (!tbl.is_connected(outc, inc)) {
                        continue;
                    }
                    __m256 delta_src = _mm256_broadcast_ss(&curr_delta[outc]);
                    const float* pw = (const float*)&W[widx];
                    __m256 w0 = _mm256_loadu_ps(pw + 0);
                    __m256 w1 = _mm256_loadu_ps(pw + 8);
                    __m256 w2 = _mm256_loadu_ps(pw + 16);
                    __m128 w3 = _mm_load_ss(pw + 24);
                    sum0 = madd256_ps(w0, delta_src, sum0);
                    sum1 = madd256_ps(w1, delta_src, sum1);
                    sum2 = madd256_ps(w2, delta_src, sum2);
                    sum3 = madd128_ss(w3, _mm256_castps256_ps128(delta_src), sum3);
                }
            }

            // *FROM
            // 1110 0000
            // 3222 2211
            // 4444 3333
            // ---- ---4
            //
            // *TO
            // ---0 0000
            // ---1 1111
            // ---2 2222
            // ---3 3333
            // ---4 4444
            dst0 = _mm256_blend_ps(
                dst0,
                sum0,
                0x1F /* 0b00011111 */
            );
            dst1 = _mm256_blend_ps(
                dst1,
                _mm256_or_ps(
                    rightShift<20>(sum0),
                    leftShift<12>(sum1)
                ),
                0x1F /* 0b00011111 */
            );
            dst2 = _mm256_blend_ps(
                dst2,
                rightShift<8>(sum1),
                0x1F /* 0b00011111 */
            );
            dst3 = _mm256_blend_ps(
                dst3,
                _mm256_or_ps(
                    rightShift<28>(sum1),
                    leftShift<4>(sum2)
                ),
                0x1F /* 0b00011111 */
            );
            dst4 = _mm256_blend_ps(
                dst4,
                _mm256_set_m128(
                    sum3,
                    _mm256_extractf128_ps(sum2, 1)
                ),
                0x1F /* 0b00011111 */
            );

            _mm256_storeu_ps(delta_dst0, dst0);
            _mm256_storeu_ps(delta_dst1, dst1);
            _mm256_storeu_ps(delta_dst2, dst2);
            _mm256_storeu_ps(delta_dst3, dst3);
            _mm256_maskstore_ps(delta_dst4, imask, dst4);
        } // for
    } else {
        for (serial_size_t inc = 0; inc < in.depth_; ++inc, pdelta_dst_org += in_padded_area) {
            for (serial_size_t outc = 0; outc < out.depth_; outc++) {
                if (!tbl.is_connected(outc, inc)) continue;

                const float* pw = &W[25 * (in.depth_ * outc + inc)];
                const float* pdelta_src = &curr_delta[out.get_index(0, 0, outc)];
                float* pdelta_dst = pdelta_dst_org;
                __m256 w0a = _mm256_maskload_ps(pw+0, imask);
                __m256 w1a = _mm256_maskload_ps(pw+5, imask);
                __m256 w2a = _mm256_maskload_ps(pw+10, imask);
                __m256 w3a = _mm256_maskload_ps(pw+15, imask);
                __m256 w4a = _mm256_maskload_ps(pw+20, imask);
                for (serial_size_t y = 0; y < out.height_; y++) {
                    float* delta_dst0 = pdelta_dst;
                    float* delta_dst1 = &pdelta_dst[in_padded.width_ * 1];
                    float* delta_dst2 = &pdelta_dst[in_padded.width_ * 2];
                    float* delta_dst3 = &pdelta_dst[in_padded.width_ * 3];
                    float* delta_dst4 = &pdelta_dst[in_padded.width_ * 4];
                    for (serial_size_t x = 0; x < out.width_; x++) {
                        __m256 delta_src = _mm256_broadcast_ss(pdelta_src + x);
                        __m256 dst0 = _mm256_loadu_ps(delta_dst0);
                        __m256 dst1 = _mm256_loadu_ps(delta_dst1);
                        __m256 dst2 = _mm256_loadu_ps(delta_dst2);
                        __m256 dst3 = _mm256_loadu_ps(delta_dst3);
                        __m256 dst4 = _mm256_maskload_ps(delta_dst4, imask);
                        dst0 = madd256_ps(w0a, delta_src, dst0);
                        dst1 = madd256_ps(w1a, delta_src, dst1);
                        dst2 = madd256_ps(w2a, delta_src, dst2);
                        dst3 = madd256_ps(w3a, delta_src, dst3);
                        dst4 = madd256_ps(w4a, delta_src, dst4);
                        _mm256_storeu_ps(delta_dst0, dst0);
                        _mm256_storeu_ps(delta_dst1, dst1);
                        _mm256_storeu_ps(delta_dst2, dst2);
                        _mm256_storeu_ps(delta_dst3, dst3);
                        _mm256_maskstore_ps(delta_dst4, imask, dst4);
                        delta_dst0 += w_stride;
                        delta_dst1 += w_stride;
                        delta_dst2 += w_stride;
                        delta_dst3 += w_stride;
                        delta_dst4 += w_stride;
                    } // for x
                    pdelta_src += out.width_;
                    pdelta_dst += h_stride2;
                } // for y
            } // for outc
        } // for inc
    }

    // accumulate dw
    if (out.width_ == 1 && out.height_ == 1) {
        const float* pprev_out = &prev_out[0];
        for (serial_size_t inc = 0; inc < in.depth_; ++inc, pprev_out += in_padded_area) {
            VECTORIZE_ALIGN(32) float floats[28];
            size_t in_padded_width = in_padded.width_;
            _mm256_store_ps(&floats[0], _mm256_loadu_ps(pprev_out + in_padded_width * 0));
            _mm256_storeu_ps(&floats[5], _mm256_loadu_ps(pprev_out + in_padded_width * 1));
            _mm256_storeu_ps(&floats[10], _mm256_loadu_ps(pprev_out + in_padded_width * 2));
            _mm256_storeu_ps(&floats[15], _mm256_loadu_ps(pprev_out + in_padded_width * 3));
            _mm256_storeu_ps(&floats[20], _mm256_maskload_ps(pprev_out + in_padded_width * 4, imask));
            __m256 prevos0 = _mm256_load_ps(&floats[0]);
            __m256 prevos1 = _mm256_load_ps(&floats[8]);
            __m256 prevos2 = _mm256_load_ps(&floats[16]);
            __m128 prevos3 = _mm_load_ss(&floats[24]);
            serial_size_t widx = 25 * inc;
            serial_size_t widx_delta = 25 * in.depth_;
            float* pdW = &dW[widx];
            for (serial_size_t outc = 0; outc < out.depth_; outc++, pdW += widx_delta) {
                if (!tbl.is_connected(outc, inc)) {
                    continue;
                }
                __m256 delta = _mm256_broadcast_ss(&curr_delta[outc]);
                __m256 w0 = _mm256_loadu_ps(pdW+0);
                __m256 w1 = _mm256_loadu_ps(pdW+8);
                __m256 w2 = _mm256_loadu_ps(pdW + 16);
                __m128 w3 = _mm_load_ss(pdW + 24);
                w0 = madd256_ps(prevos0, delta, w0);
                w1 = madd256_ps(prevos1, delta, w1);
                w2 = madd256_ps(prevos2, delta, w2);
                w3 = madd128_ss(prevos3, _mm256_castps256_ps128(delta), w3);
                _mm256_storeu_ps(pdW + 0, w0);
                _mm256_storeu_ps(pdW + 8, w1);
                _mm256_storeu_ps(pdW+16, w2);
                _mm_store_ss(pdW+24, w3);
            }
        }
    } else {
        // prepare load-mask beforehand
        const size_t nblocks = out.width_ >> 3;
        static const int32_t masks[] = {
            -1, -1, -1, -1,
            -1, -1, -1, -1,
            0, 0, 0, 0,
            0, 0, 0, 0,
        };
        const size_t remainder = out.width_ & 7;
        __m256i mask = _mm256_loadu_si256((const __m256i*)(masks + 8 - remainder));
        auto& weight = params.weight;
        for (serial_size_t inc = 0; inc < in.depth_; ++inc) {
            for (serial_size_t outc = 0; outc < out.depth_; outc++) {

                if (!tbl.is_connected(outc, inc)) continue;
                const float* delta = &curr_delta[out.get_index(0, 0, outc)];

                serial_size_t widx = weight.get_index(0, 0, in.depth_ * outc + inc);
                for (serial_size_t wy = 0; wy < 5 /* weight.height_ */; wy++) {
                    for (serial_size_t wx = 0; wx < 5 /* weight.width_ */; wx++) {
                        const float* prevo = &prev_out[in_padded.get_index(wx, wy, inc)];

                        if (w_stride > 1) {
                            float_t dst = float_t(0);

                            for (serial_size_t y = 0; y < params.out.height_; y++) {
                                serial_size_t prevo_idx = y * params.in_padded.width_ * params.h_stride;
                                serial_size_t delta_idx = y * params.out.width_;

                                for (serial_size_t x = 0; x < params.out.width_; x++) {
                                    dst += prevo[prevo_idx + x * params.w_stride] * delta[delta_idx + x];
                                }
                            }
                            dW[widx] += dst;
                        }
                        else {
                            __m128 prev_sum = _mm_load_ss(&dW[widx]);
                            __m256 sum0 = _mm256_setzero_ps();
                            __m256 sum1 = _mm256_setzero_ps();
                            for (serial_size_t y = 0; y < out.height_; y++) {
                                // vectorize::dot
                                const float* pa = prevo + y * in_padded.width_ * params.h_stride;
                                const float* pb = delta + y * out.width_;
                                for (size_t i = 0; i < nblocks; ++i) {
                                    __m256 a = _mm256_loadu_ps(pa + 8 * i);
                                    __m256 b = _mm256_loadu_ps(pb + 8 * i);
                                    sum0 = madd256_ps(a, b, sum0);
                                }
                                if (remainder) {
                                    __m256 a = _mm256_maskload_ps(pa + 8 * nblocks, mask);
                                    __m256 b = _mm256_maskload_ps(pb + 8 * nblocks, mask);
                                    sum1 = madd256_ps(a, b, sum1);
                                }
                            }
                            sum1 = _mm256_and_ps(sum1, _mm256_castsi256_ps(mask));
                            __m256 sum = _mm256_add_ps(sum0, sum1);
                            _mm_store_ss(&dW[widx], _mm_add_ps(prev_sum, hsum256_ps(sum)));
                        }
                        ++widx;
                    }
                }
            }
        }
    }

    // accumulate db
    if (params.has_bias) {
        //fvec_t& db = *in_grad[2];
        
        if (out.width_ == 1 && out.height_ == 1) {
            for (serial_size_t outc = 0; outc < out.depth_; outc++) {
                db[outc] += curr_delta[outc];
            }
        } else {
            for (serial_size_t outc = 0; outc < out.depth_; outc++) {
                const float *delta = &curr_delta[out.get_index(0, 0, outc)];
                db[outc] += std::accumulate(delta, delta + out.width_ * out.height_, float(0));
            }
        }
    }
} // avx_conv2d_5x5_back_kernel float ver

// double ver
template <typename Allocator>
void avx_conv2d_5x5_back_kernel(const core::conv_params& params,
                                const std::vector<std::vector<double, Allocator>>& prev_out,
                                const std::vector<double, Allocator>& W,
                                std::vector<std::vector<double, Allocator>>&       dW,
                                std::vector<std::vector<double, Allocator>>&       db,
                                std::vector<std::vector<double, Allocator>>&       curr_delta,
                                std::vector<std::vector<double, Allocator>>&       prev_delta) {
    // backward-pass fallbacks to tiny-backend at float_t == double
    conv2d_op_internal(prev_out, W, dW, db, curr_delta, prev_delta, params, true);
}

// float ver
template <typename Allocator>
void avx_conv2d_5x5_back_kernel(const core::conv_params& params,
                                const std::vector<std::vector<float, Allocator>>& prev_out,
                                const std::vector<float, Allocator>& W,
                                std::vector<std::vector<float, Allocator>>&       dW,
                                std::vector<std::vector<float, Allocator>>&       db,
                                std::vector<std::vector<float, Allocator>>&       curr_delta,
                                std::vector<std::vector<float, Allocator>>&       prev_delta) {
    for_i(prev_out.size(), [&](int sample) {
        avx_conv2d_5x5_back_kernel_one(params, prev_out[sample], W, dW[sample], db[sample],
            curr_delta[sample], &prev_delta[sample]);
    });
} 


#endif // CNN_USE_AVX

inline void
conv2d_grad_op_avx(const tensor_t&        prev_out,
                   const vec_t&                  W,
                   tensor_t&                    dW,
                   tensor_t&                    db,
                   tensor_t&            curr_delta,
                   tensor_t&            prev_delta,
                   const core::conv_params& params,
                   const bool    layer_parallelize) {
#ifdef CNN_USE_AVX
    if (params.weight.height_ == 5 && params.weight.width_ == 5) {
        avx_conv2d_5x5_back_kernel(params, prev_out, W, dW, db, curr_delta, prev_delta);
        return;
    }
#endif

    conv2d_op_internal(prev_out, W, dW, db, curr_delta,
                       prev_delta, params, layer_parallelize);
}

}  // namespace kernels
}  // namespace tiny_dnn
