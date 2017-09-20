/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <vector>
#include "tiny_dnn/core/kernels/conv2d_op_internal.h"
#include "tiny_dnn/core/params/conv_params.h"

#ifdef CNN_USE_AVX
#include "tiny_dnn/core/kernels/avx_kernel_common.h"
#endif

namespace tiny_dnn {
namespace kernels {

#ifdef CNN_USE_AVX

// float ver
template <typename Allocator>
inline void accumulate_db(const index3d<size_t> &out,
                          const std::vector<float, Allocator> &curr_delta,
                          std::vector<float, Allocator> &db) {
  if (out.width_ == 1 && out.height_ == 1) {
    size_t nblocks = out.depth_ / 8;
    for (size_t i = 0; i < nblocks; ++i) {
      _mm256_storeu_ps(&db[i * 8],
                       _mm256_add_ps(_mm256_loadu_ps(&db[i * 8]),
                                     _mm256_loadu_ps(&curr_delta[i * 8])));
    }
    for (size_t outc = nblocks * 8; outc < out.depth_; ++outc) {
      db[outc] += curr_delta[outc];
    }
  } else {
    auto area        = out.area();
    size_t n8        = area / 64;
    size_t n4        = (area % 64) / 32;
    size_t n2        = (area % 32) / 16;
    size_t n1        = (area % 16) / 8;
    size_t remainder = area & 7;
    // prepare load-mask beforehand
    static const int32_t masks[] = {
      -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    __m256i mask = _mm256_loadu_si256((const __m256i *)(masks + 8 - remainder));
    for (size_t outc = 0; outc < out.depth_; ++outc) {
      size_t idx         = out.get_index(0, 0, outc);
      const float *delta = &curr_delta[idx];
      __m256 sum0        = _mm256_setzero_ps();
      __m256 sum1        = _mm256_setzero_ps();
      __m256 sum2        = _mm256_setzero_ps();
      __m256 sum3        = _mm256_setzero_ps();
      __m256 sum4        = _mm256_setzero_ps();
      __m256 sum5        = _mm256_setzero_ps();
      __m256 sum6        = _mm256_setzero_ps();
      __m256 sum7        = _mm256_setzero_ps();
      for (size_t i = 0; i < n8; ++i) {
        sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(delta + i * 64 + 0));
        sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(delta + i * 64 + 8));
        sum2 = _mm256_add_ps(sum2, _mm256_loadu_ps(delta + i * 64 + 16));
        sum3 = _mm256_add_ps(sum3, _mm256_loadu_ps(delta + i * 64 + 24));
        sum4 = _mm256_add_ps(sum4, _mm256_loadu_ps(delta + i * 64 + 32));
        sum5 = _mm256_add_ps(sum5, _mm256_loadu_ps(delta + i * 64 + 40));
        sum6 = _mm256_add_ps(sum6, _mm256_loadu_ps(delta + i * 64 + 48));
        sum7 = _mm256_add_ps(sum7, _mm256_loadu_ps(delta + i * 64 + 56));
      }
      delta += n8 * 64;
      if (n4) {
        sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(delta + 0));
        sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(delta + 8));
        sum2 = _mm256_add_ps(sum2, _mm256_loadu_ps(delta + 16));
        sum3 = _mm256_add_ps(sum3, _mm256_loadu_ps(delta + 24));
        delta += 32;
      }
      if (n2) {
        sum4 = _mm256_add_ps(sum4, _mm256_loadu_ps(delta + 0));
        sum5 = _mm256_add_ps(sum5, _mm256_loadu_ps(delta + 8));
        delta += 16;
      }
      if (n1) {
        sum6 = _mm256_add_ps(sum6, _mm256_loadu_ps(delta));
        delta += 8;
      }
      sum0 = _mm256_add_ps(sum0, sum1);
      sum2 = _mm256_add_ps(sum2, sum3);
      sum4 = _mm256_add_ps(sum4, sum5);
      sum6 = _mm256_add_ps(sum6, sum7);
      sum1 = _mm256_maskload_ps(delta, mask);
      sum0 = _mm256_add_ps(sum0, sum2);
      sum4 = _mm256_add_ps(sum4, sum6);
      sum0 = _mm256_add_ps(sum0, sum4);
      sum0 = _mm256_add_ps(sum0, sum1);
      db[outc] += _mm_cvtss_f32(hsum256_ps(sum0));
    }
  }
}  // accumulate_db

// float ver
template <typename Allocator>
inline void accumulate_dw(const core::conv_params &params,
                          const std::vector<float, Allocator> &prev_out,
                          const std::vector<float, Allocator> &curr_delta,
                          std::vector<float, Allocator> &dW,
                          std::vector<float, Allocator> &db) {
  CNN_UNREFERENCED_PARAMETER(db);
  auto &in                    = params.in;
  auto &out                   = params.out;
  auto &in_padded             = params.in_padded;
  auto &tbl                   = params.tbl;
  auto w_stride               = params.w_stride;
  const size_t in_padded_area = in_padded.area();
  static const __m256i imask  = _mm256_setr_epi32(-1, -1, -1, -1, -1, 0, 0, 0);

  if (out.width_ == 1 && out.height_ == 1) {
    const float *pprev_out = &prev_out[0];
    alignas(32) float floats[28];
    for (size_t inc = 0; inc < in.depth_; ++inc, pprev_out += in_padded_area) {
      size_t in_padded_width = in_padded.width_;
      _mm256_store_ps(&floats[0],
                      _mm256_loadu_ps(pprev_out + in_padded_width * 0));
      _mm256_storeu_ps(&floats[5],
                       _mm256_loadu_ps(pprev_out + in_padded_width * 1));
      _mm256_storeu_ps(&floats[10],
                       _mm256_loadu_ps(pprev_out + in_padded_width * 2));
      _mm256_storeu_ps(&floats[15],
                       _mm256_loadu_ps(pprev_out + in_padded_width * 3));
      _mm256_storeu_ps(&floats[20], _mm256_maskload_ps(
                                      pprev_out + in_padded_width * 4, imask));
      __m256 prevos0    = _mm256_load_ps(&floats[0]);
      __m256 prevos1    = _mm256_load_ps(&floats[8]);
      __m256 prevos2    = _mm256_load_ps(&floats[16]);
      __m128 prevos3    = _mm_load_ss(&floats[24]);
      size_t widx       = 25 * inc;
      size_t widx_delta = 25 * in.depth_;
      float *pdW        = &dW[widx];
      for (size_t outc = 0; outc < out.depth_; outc++, pdW += widx_delta) {
        if (!tbl.is_connected(outc, inc)) {
          continue;
        }
        __m256 delta = _mm256_broadcast_ss(&curr_delta[outc]);
        __m256 w0    = _mm256_loadu_ps(pdW + 0);
        __m256 w1    = _mm256_loadu_ps(pdW + 8);
        __m256 w2    = _mm256_loadu_ps(pdW + 16);
        __m128 w3    = _mm_load_ss(pdW + 24);
        w0           = madd256_ps(prevos0, delta, w0);
        w1           = madd256_ps(prevos1, delta, w1);
        w2           = madd256_ps(prevos2, delta, w2);
        w3           = madd128_ss(prevos3, _mm256_castps256_ps128(delta), w3);
        _mm256_storeu_ps(pdW + 0, w0);
        _mm256_storeu_ps(pdW + 8, w1);
        _mm256_storeu_ps(pdW + 16, w2);
        _mm_store_ss(pdW + 24, w3);
      }
    }
  } else {
    // prepare load-mask beforehand
    const size_t nblocks         = out.width_ >> 3;
    static const int32_t masks[] = {
      -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    const size_t remainder = out.width_ & 7;
    __m256i mask = _mm256_loadu_si256((const __m256i *)(masks + 8 - remainder));
    auto &weight = params.weight;
    size_t prevo_delta      = in_padded.width_ * params.h_stride;
    const size_t out_width  = out.width_;
    const size_t out_height = out.height_;
    assert(1 < out_width);
    assert(1 < out_height);
    __m256 sum0, sum1, sum2, sum3, sum4;
    if (w_stride > 1) {
      for (size_t inc = 0; inc < in.depth_; ++inc) {
        for (size_t outc = 0; outc < out.depth_; ++outc) {
          const float *delta = &curr_delta[out.get_index(0, 0, outc)];
          if (!tbl.is_connected(outc, inc)) {
            continue;
          }
          size_t widx = weight.get_index(0, 0, in.depth_ * outc + inc);
          // weight.height_
          for (size_t wy = 0; wy < 5; ++wy) {
            // weight.width_
            for (size_t wx = 0; wx < 5; ++wx, ++widx) {
              size_t prev_out_idx = in_padded.get_index(wx, wy, inc);
              const float *prevo  = &prev_out[prev_out_idx];
              float_t dst{0};
              for (size_t y = 0, prevo_idx = 0, delta_idx = 0; y < out_height;
                   ++y, prevo_idx += prevo_delta, delta_idx += out_width) {
                for (size_t x = 0; x < out_width; ++x) {
                  dst += prevo[prevo_idx + x * params.w_stride] *
                         delta[delta_idx + x];
                }
              }
              dW[widx] += dst;
            }  // for wx
          }    // for wy
        }      // for outc
      }        // for inc
    } else if (nblocks == 1 && remainder != 0) {
      for (size_t inc = 0; inc < in.depth_; ++inc) {
        for (size_t outc = 0; outc < out.depth_; ++outc) {
          if (!tbl.is_connected(outc, inc)) {
            continue;
          }
          const float *delta = &curr_delta[out.get_index(0, 0, outc)];
          size_t widx        = weight.get_index(0, 0, in.depth_ * outc + inc);
          float *pdw         = &dW[widx];
          // weight.height_
          for (size_t wy = 0; wy < 5; ++wy) {
            size_t prev_out_idx = in_padded.get_index(0, wy, inc);
            const float *pa     = &prev_out[prev_out_idx];
            const float *pb     = delta;
            // y = 0
            sum0 = sum1 = sum2 = sum3 = sum4 = _mm256_setzero_ps();
            for (size_t y = 0; y < out_height; ++y) {
              // vectorize::dot
              __m256 a0 = _mm256_loadu_ps(pa + 0);
              __m256 a1 = _mm256_loadu_ps(pa + 1);
              __m256 a2 = _mm256_loadu_ps(pa + 2);
              __m256 a3 = _mm256_loadu_ps(pa + 3);
              __m256 a4 = _mm256_loadu_ps(pa + 4);
              __m256 b  = _mm256_loadu_ps(pb);
              sum0      = madd256_ps(a0, b, sum0);
              sum1      = madd256_ps(a1, b, sum1);
              sum2      = madd256_ps(a2, b, sum2);
              sum3      = madd256_ps(a3, b, sum3);
              sum4      = madd256_ps(a4, b, sum4);
              a0        = _mm256_maskload_ps(pa + 010, mask);
              a1        = _mm256_maskload_ps(pa + 011, mask);
              a2        = _mm256_maskload_ps(pa + 012, mask);
              a3        = _mm256_maskload_ps(pa + 013, mask);
              a4        = _mm256_maskload_ps(pa + 014, mask);
              b         = _mm256_maskload_ps(pb + 010, mask);
              sum0      = madd256_ps(a0, b, sum0);
              sum1      = madd256_ps(a1, b, sum1);
              sum2      = madd256_ps(a2, b, sum2);
              sum3      = madd256_ps(a3, b, sum3);
              sum4      = madd256_ps(a4, b, sum4);
              pa += prevo_delta;
              pb += out_width;
            }
            _mm_storeu_ps(pdw + wy * 5,
                          _mm_add_ps(_mm_loadu_ps(pdw + wy * 5),
                                     hsum4x256_ps(sum0, sum1, sum2, sum3)));
            _mm_store_ss(
              pdw + wy * 5 + 4,
              _mm_add_ss(_mm_load_ss(pdw + wy * 5 + 4), hsum256_ps(sum4)));
          }  // for wy
        }    // for outc
      }      // for inc
    } else if (nblocks > 1 && remainder != 0) {
      for (size_t inc = 0; inc < in.depth_; ++inc) {
        for (size_t outc = 0; outc < out.depth_; ++outc) {
          if (!tbl.is_connected(outc, inc)) {
            continue;
          }
          const float *delta = &curr_delta[out.get_index(0, 0, outc)];
          size_t widx        = weight.get_index(0, 0, in.depth_ * outc + inc);
          float *pdw         = &dW[widx];
          // weight.height_
          for (size_t wy = 0; wy < 5; ++wy) {
            size_t prev_out_idx = in_padded.get_index(0, wy, inc);
            const float *pa     = &prev_out[prev_out_idx];
            const float *pb     = delta;
            sum0 = sum1 = sum2 = sum3 = sum4 = _mm256_setzero_ps();
            for (size_t y = 0; y < out_height;
                 ++y, pa += prevo_delta, pb += out_width) {
              // vectorize::dot
              __m256 a0 = _mm256_loadu_ps(pa + 0);
              __m256 a1 = _mm256_loadu_ps(pa + 1);
              __m256 a2 = _mm256_loadu_ps(pa + 2);
              __m256 a3 = _mm256_loadu_ps(pa + 3);
              __m256 a4 = _mm256_loadu_ps(pa + 4);
              __m256 b  = _mm256_loadu_ps(pb);
              sum0      = madd256_ps(a0, b, sum0);
              sum1      = madd256_ps(a1, b, sum1);
              sum2      = madd256_ps(a2, b, sum2);
              sum3      = madd256_ps(a3, b, sum3);
              sum4      = madd256_ps(a4, b, sum4);
              a0        = _mm256_loadu_ps(pa + 010);
              a1        = _mm256_loadu_ps(pa + 011);
              a2        = _mm256_loadu_ps(pa + 012);
              a3        = _mm256_loadu_ps(pa + 013);
              a4        = _mm256_loadu_ps(pa + 014);
              b         = _mm256_loadu_ps(pb + 010);
              sum0      = madd256_ps(a0, b, sum0);
              sum1      = madd256_ps(a1, b, sum1);
              sum2      = madd256_ps(a2, b, sum2);
              sum3      = madd256_ps(a3, b, sum3);
              sum4      = madd256_ps(a4, b, sum4);
              for (size_t i = 2; i < nblocks; ++i) {
                a0   = _mm256_loadu_ps(pa + 8 * i + 0);
                a1   = _mm256_loadu_ps(pa + 8 * i + 1);
                a2   = _mm256_loadu_ps(pa + 8 * i + 2);
                a3   = _mm256_loadu_ps(pa + 8 * i + 3);
                a4   = _mm256_loadu_ps(pa + 8 * i + 4);
                b    = _mm256_loadu_ps(pb + 8 * i);
                sum0 = madd256_ps(a0, b, sum0);
                sum1 = madd256_ps(a1, b, sum1);
                sum2 = madd256_ps(a2, b, sum2);
                sum3 = madd256_ps(a3, b, sum3);
                sum4 = madd256_ps(a4, b, sum4);
              }
              a0   = _mm256_maskload_ps(pa + 8 * nblocks + 0, mask);
              a1   = _mm256_maskload_ps(pa + 8 * nblocks + 1, mask);
              a2   = _mm256_maskload_ps(pa + 8 * nblocks + 2, mask);
              a3   = _mm256_maskload_ps(pa + 8 * nblocks + 3, mask);
              a4   = _mm256_maskload_ps(pa + 8 * nblocks + 4, mask);
              b    = _mm256_maskload_ps(pb + 8 * nblocks, mask);
              sum0 = madd256_ps(a0, b, sum0);
              sum1 = madd256_ps(a1, b, sum1);
              sum2 = madd256_ps(a2, b, sum2);
              sum3 = madd256_ps(a3, b, sum3);
              sum4 = madd256_ps(a4, b, sum4);
            }
            _mm_storeu_ps(pdw + wy * 5,
                          _mm_add_ps(_mm_loadu_ps(pdw + wy * 5),
                                     hsum4x256_ps(sum0, sum1, sum2, sum3)));
            _mm_store_ss(
              pdw + wy * 5 + 4,
              _mm_add_ss(_mm_load_ss(pdw + wy * 5 + 4), hsum256_ps(sum4)));
          }  // for wy
        }    // for outc
      }      // for inc
    } else if (nblocks == 0) {
      assert(remainder != 0);
      for (size_t inc = 0; inc < in.depth_; ++inc) {
        for (size_t outc = 0; outc < out.depth_; ++outc) {
          if (!tbl.is_connected(outc, inc)) {
            continue;
          }
          const float *delta = &curr_delta[out.get_index(0, 0, outc)];
          size_t widx        = weight.get_index(0, 0, in.depth_ * outc + inc);
          float *pdw         = &dW[widx];
          // weight.height_
          for (size_t wy = 0; wy < 5; ++wy) {
            size_t prev_out_idx = in_padded.get_index(0, wy, inc);
            const float *pa     = &prev_out[prev_out_idx];
            const float *pb     = delta;
            // vectorize::dot
            sum0 = sum1 = sum2 = sum3 = sum4 = _mm256_setzero_ps();
            for (size_t y = 0; y < out_height; ++y) {
              // vectorize::dot
              __m256 a0 = _mm256_maskload_ps(pa + 0, mask);
              __m256 a1 = _mm256_maskload_ps(pa + 1, mask);
              __m256 a2 = _mm256_maskload_ps(pa + 2, mask);
              __m256 a3 = _mm256_maskload_ps(pa + 3, mask);
              __m256 a4 = _mm256_maskload_ps(pa + 4, mask);
              __m256 b  = _mm256_maskload_ps(pb, mask);
              sum0      = madd256_ps(a0, b, sum0);
              sum1      = madd256_ps(a1, b, sum1);
              sum2      = madd256_ps(a2, b, sum2);
              sum3      = madd256_ps(a3, b, sum3);
              sum4      = madd256_ps(a4, b, sum4);
              pa += prevo_delta;
              pb += out_width;
            }
            _mm_storeu_ps(pdw + wy * 5,
                          _mm_add_ps(_mm_loadu_ps(pdw + wy * 5),
                                     hsum4x256_ps(sum0, sum1, sum2, sum3)));
            _mm_store_ss(
              pdw + wy * 5 + 4,
              _mm_add_ss(_mm_load_ss(pdw + wy * 5 + 4), hsum256_ps(sum4)));
          }  // for wy
        }    // for outc
      }      // for inc
    } else if (nblocks == 1) {
      assert(remainder == 0);
      for (size_t inc = 0; inc < in.depth_; ++inc) {
        for (size_t outc = 0; outc < out.depth_; ++outc) {
          if (!tbl.is_connected(outc, inc)) {
            continue;
          }
          const float *delta = &curr_delta[out.get_index(0, 0, outc)];
          size_t widx        = weight.get_index(0, 0, in.depth_ * outc + inc);
          float *pdw         = &dW[widx];
          // weight.height_
          for (size_t wy = 0; wy < 5; ++wy) {
            size_t prev_out_idx = in_padded.get_index(0, wy, inc);
            const float *pa     = &prev_out[prev_out_idx];
            const float *pb     = delta;
            // vectorize::dot
            sum0 = sum1 = sum2 = sum3 = sum4 = _mm256_setzero_ps();
            for (size_t y = 0; y < out_height; ++y) {
              // vectorize::dot
              __m256 a0 = _mm256_loadu_ps(pa + 0);
              __m256 a1 = _mm256_loadu_ps(pa + 1);
              __m256 a2 = _mm256_loadu_ps(pa + 2);
              __m256 a3 = _mm256_loadu_ps(pa + 3);
              __m256 a4 = _mm256_loadu_ps(pa + 4);
              __m256 b  = _mm256_loadu_ps(pb);
              sum0      = madd256_ps(a0, b, sum0);
              sum1      = madd256_ps(a1, b, sum1);
              sum2      = madd256_ps(a2, b, sum2);
              sum3      = madd256_ps(a3, b, sum3);
              sum4      = madd256_ps(a4, b, sum4);
              pa += prevo_delta;
              pb += out_width;
            }
            _mm_storeu_ps(pdw + wy * 5,
                          _mm_add_ps(_mm_loadu_ps(pdw + wy * 5),
                                     hsum4x256_ps(sum0, sum1, sum2, sum3)));
            _mm_store_ss(
              pdw + wy * 5 + 4,
              _mm_add_ss(_mm_load_ss(pdw + wy * 5 + 4), hsum256_ps(sum4)));
          }  // for wy
        }    // for outc
      }      // for inc
    } else {
      assert(nblocks > 1);
      assert(remainder == 0);
      for (size_t inc = 0; inc < in.depth_; ++inc) {
        for (size_t outc = 0; outc < out.depth_; ++outc) {
          if (!tbl.is_connected(outc, inc)) {
            continue;
          }
          const float *delta = &curr_delta[out.get_index(0, 0, outc)];
          size_t widx        = weight.get_index(0, 0, in.depth_ * outc + inc);
          float *pdw         = &dW[widx];
          // weight.height_
          for (size_t wy = 0; wy < 5; ++wy) {
            size_t prev_out_idx = in_padded.get_index(0, wy, inc);
            const float *pa     = &prev_out[prev_out_idx];
            const float *pb     = delta;
            // vectorize::dot
            sum0 = sum1 = sum2 = sum3 = sum4 = _mm256_setzero_ps();
            for (size_t y = 0; y < out_height; ++y) {
              // vectorize::dot
              __m256 a0 = _mm256_loadu_ps(pa + 0);
              __m256 a1 = _mm256_loadu_ps(pa + 1);
              __m256 a2 = _mm256_loadu_ps(pa + 2);
              __m256 a3 = _mm256_loadu_ps(pa + 3);
              __m256 a4 = _mm256_loadu_ps(pa + 4);
              __m256 b  = _mm256_loadu_ps(pb);
              sum0      = madd256_ps(a0, b, sum0);
              sum1      = madd256_ps(a1, b, sum1);
              sum2      = madd256_ps(a2, b, sum2);
              sum3      = madd256_ps(a3, b, sum3);
              sum4      = madd256_ps(a4, b, sum4);
              a0        = _mm256_loadu_ps(pa + 010);
              a1        = _mm256_loadu_ps(pa + 011);
              a2        = _mm256_loadu_ps(pa + 012);
              a3        = _mm256_loadu_ps(pa + 013);
              a4        = _mm256_loadu_ps(pa + 014);
              b         = _mm256_loadu_ps(pb + 010);
              sum0      = madd256_ps(a0, b, sum0);
              sum1      = madd256_ps(a1, b, sum1);
              sum2      = madd256_ps(a2, b, sum2);
              sum3      = madd256_ps(a3, b, sum3);
              sum4      = madd256_ps(a4, b, sum4);
              for (size_t i = 2; i < nblocks; ++i) {
                a0   = _mm256_loadu_ps(pa + 8 * i + 0);
                a1   = _mm256_loadu_ps(pa + 8 * i + 1);
                a2   = _mm256_loadu_ps(pa + 8 * i + 2);
                a3   = _mm256_loadu_ps(pa + 8 * i + 3);
                a4   = _mm256_loadu_ps(pa + 8 * i + 4);
                b    = _mm256_loadu_ps(pb + 8 * i);
                sum0 = madd256_ps(a0, b, sum0);
                sum1 = madd256_ps(a1, b, sum1);
                sum2 = madd256_ps(a2, b, sum2);
                sum3 = madd256_ps(a3, b, sum3);
                sum4 = madd256_ps(a4, b, sum4);
              }
              pa += prevo_delta;
              pb += out_width;
            }
            _mm_storeu_ps(pdw + wy * 5,
                          _mm_add_ps(_mm_loadu_ps(pdw + wy * 5),
                                     hsum4x256_ps(sum0, sum1, sum2, sum3)));
            _mm_store_ss(
              pdw + wy * 5 + 4,
              _mm_add_ss(_mm_load_ss(pdw + wy * 5 + 4), hsum256_ps(sum4)));
          }  // for wy
        }    // for outc
      }      // for inc
    }
  }  // else
}  // accumulate_dw

// float ver
template <typename Allocator>
void avx_conv2d_5x5_back_kernel_one(
  const core::conv_params &params,
  const std::vector<float, Allocator> &prev_out,
  const std::vector<float, Allocator> &W,
  std::vector<float, Allocator> &dW,
  std::vector<float, Allocator> &db,
  std::vector<float, Allocator> &curr_delta,
  std::vector<float, Allocator> *prev_delta) {
  auto &in                    = params.in;
  auto &out                   = params.out;
  auto &in_padded             = params.in_padded;
  auto &tbl                   = params.tbl;
  auto w_stride               = params.w_stride;
  const size_t in_padded_area = in_padded.area();
  float *pdelta_dst_org       = &(*prev_delta)[0];
  const size_t h_stride2      = params.h_stride * in_padded.width_;
  static const __m256i imask  = _mm256_setr_epi32(-1, -1, -1, -1, -1, 0, 0, 0);
  static const __m256 mask =
    _mm256_castsi256_ps(_mm256_setr_epi32(-1, -1, -1, -1, -1, 0, 0, 0));
  const size_t out_width  = out.width_;
  const size_t out_height = out.height_;
  // propagate delta to previous layer
  if (w_stride == 1 && out_width >= 4) {
    const size_t nblocks = out_width / 4;
    if (out_width % 4) {
      for (size_t inc = 0; inc < in.depth_;
           ++inc, pdelta_dst_org += in_padded_area) {
        for (size_t outc = 0; outc < out.depth_; ++outc) {
          if (!tbl.is_connected(outc, inc)) {
            continue;
          }
          const float *pw         = &W[25 * (in.depth_ * outc + inc)];
          const float *pdelta_src = &curr_delta[out.get_index(0, 0, outc)];
          float *pdelta_dst       = pdelta_dst_org;
          __m256 w0a = _mm256_and_ps(_mm256_loadu_ps(pw + 0), mask);
          __m256 w1a = _mm256_and_ps(_mm256_loadu_ps(pw + 5), mask);
          __m256 w2a = _mm256_and_ps(_mm256_loadu_ps(pw + 10), mask);
          __m256 w3a = _mm256_and_ps(_mm256_loadu_ps(pw + 15), mask);
          __m256 w4a = _mm256_maskload_ps(pw + 20, imask);
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
          for (size_t y = 0; y < out_height;
               ++y, pdelta_src += out_width, pdelta_dst += h_stride2) {
            float *delta_dst0 = pdelta_dst;
            float *delta_dst1 = &pdelta_dst[in_padded.width_ * 1];
            float *delta_dst2 = &pdelta_dst[in_padded.width_ * 2];
            float *delta_dst3 = &pdelta_dst[in_padded.width_ * 3];
            float *delta_dst4 = &pdelta_dst[in_padded.width_ * 4];
            for (size_t n = 0; n < nblocks; ++n) {
              __m256 delta_src =
                _mm256_broadcast_ps((const __m128 *)(pdelta_src + n * 4));
              __m256 dst0 = _mm256_loadu_ps(delta_dst0 + 4 * n);
              __m256 dst1 = _mm256_loadu_ps(delta_dst1 + 4 * n);
              __m256 dst2 = _mm256_loadu_ps(delta_dst2 + 4 * n);
              __m256 dst3 = _mm256_loadu_ps(delta_dst3 + 4 * n);
              __m256 dst4 = _mm256_loadu_ps(delta_dst4 + 4 * n);
              __m256 delta_src0 =
                _mm256_permute_ps(delta_src, _MM_SHUFFLE(0, 0, 0, 0));
              __m256 delta_src1 =
                _mm256_permute_ps(delta_src, _MM_SHUFFLE(1, 1, 1, 1));
              __m256 delta_src2 =
                _mm256_permute_ps(delta_src, _MM_SHUFFLE(2, 2, 2, 2));
              __m256 delta_src3 =
                _mm256_permute_ps(delta_src, _MM_SHUFFLE(3, 3, 3, 3));
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
            }  // for nblocks
            for (size_t x = nblocks * 4; x < out_width; ++x) {
              __m256 delta_src = _mm256_broadcast_ss(pdelta_src + x);
              __m256 dst0      = _mm256_loadu_ps(delta_dst0 + x);
              __m256 dst1      = _mm256_loadu_ps(delta_dst1 + x);
              __m256 dst2      = _mm256_loadu_ps(delta_dst2 + x);
              __m256 dst3      = _mm256_loadu_ps(delta_dst3 + x);
              __m256 dst4      = _mm256_maskload_ps(delta_dst4 + x, imask);
              dst0             = madd256_ps(w0a, delta_src, dst0);
              dst1             = madd256_ps(w1a, delta_src, dst1);
              dst2             = madd256_ps(w2a, delta_src, dst2);
              dst3             = madd256_ps(w3a, delta_src, dst3);
              dst4             = madd256_ps(w4a, delta_src, dst4);
              _mm256_maskstore_ps(delta_dst0 + x, imask, dst0);
              _mm256_maskstore_ps(delta_dst1 + x, imask, dst1);
              _mm256_maskstore_ps(delta_dst2 + x, imask, dst2);
              _mm256_maskstore_ps(delta_dst3 + x, imask, dst3);
              _mm256_maskstore_ps(delta_dst4 + x, imask, dst4);
            }  // for x
          }    // for out_height
        }      // for out.depth_
      }        // for in.depth_
    } else {
      for (size_t inc = 0; inc < in.depth_;
           ++inc, pdelta_dst_org += in_padded_area) {
        for (size_t outc = 0; outc < out.depth_; ++outc) {
          if (!tbl.is_connected(outc, inc)) {
            continue;
          }
          const float *pw         = &W[25 * (in.depth_ * outc + inc)];
          const float *pdelta_src = &curr_delta[out.get_index(0, 0, outc)];
          float *pdelta_dst       = pdelta_dst_org;
          __m256 w0a = _mm256_and_ps(_mm256_loadu_ps(pw + 0), mask);
          __m256 w1a = _mm256_and_ps(_mm256_loadu_ps(pw + 5), mask);
          __m256 w2a = _mm256_and_ps(_mm256_loadu_ps(pw + 10), mask);
          __m256 w3a = _mm256_and_ps(_mm256_loadu_ps(pw + 15), mask);
          __m256 w4a = _mm256_maskload_ps(pw + 20, imask);
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
          size_t y   = 0;
          do {
            float *delta_dst0 = pdelta_dst;
            float *delta_dst1 = &pdelta_dst[in_padded.width_ * 1];
            float *delta_dst2 = &pdelta_dst[in_padded.width_ * 2];
            float *delta_dst3 = &pdelta_dst[in_padded.width_ * 3];
            float *delta_dst4 = &pdelta_dst[in_padded.width_ * 4];
            size_t n          = 0;
            do {
              __m256 delta_src =
                _mm256_broadcast_ps((const __m128 *)(pdelta_src + n * 4));
              __m256 dst0 = _mm256_loadu_ps(delta_dst0 + 4 * n);
              __m256 dst1 = _mm256_loadu_ps(delta_dst1 + 4 * n);
              __m256 dst2 = _mm256_loadu_ps(delta_dst2 + 4 * n);
              __m256 dst3 = _mm256_loadu_ps(delta_dst3 + 4 * n);
              __m256 dst4 = _mm256_loadu_ps(delta_dst4 + 4 * n);
              __m256 delta_src0 =
                _mm256_permute_ps(delta_src, _MM_SHUFFLE(0, 0, 0, 0));
              __m256 delta_src1 =
                _mm256_permute_ps(delta_src, _MM_SHUFFLE(1, 1, 1, 1));
              __m256 delta_src2 =
                _mm256_permute_ps(delta_src, _MM_SHUFFLE(2, 2, 2, 2));
              __m256 delta_src3 =
                _mm256_permute_ps(delta_src, _MM_SHUFFLE(3, 3, 3, 3));
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
              ++n;
            } while (n < nblocks);
            ++y;
            pdelta_src += out_width;
            pdelta_dst += h_stride2;
          } while (y < out_height);
        }  // for out.depth_
      }    // for in.depth_
    }
  } else if (out_height == 1 && out_width == 1) {
    for (size_t inc = 0; inc < in.depth_;
         ++inc, pdelta_dst_org += in_padded_area) {
      __m256 sum0 = _mm256_setzero_ps();
      __m256 sum1 = _mm256_setzero_ps();
      __m256 sum2 = _mm256_setzero_ps();
      __m128 sum3 = _mm_setzero_ps();

      size_t widx  = 25 * inc;
      size_t wstep = 25 * in.depth_;
      __m256 delta_src;
      if (tbl.is_empty()) {
        for (size_t outc = 0; outc < out.depth_; ++outc, widx += wstep) {
          delta_src       = _mm256_broadcast_ss(&curr_delta[outc]);
          const float *pw = (const float *)&W[widx];
          __m256 w0       = _mm256_loadu_ps(pw + 0);
          __m256 w1       = _mm256_loadu_ps(pw + 8);
          __m256 w2       = _mm256_loadu_ps(pw + 16);
          __m128 w3       = _mm_load_ss(pw + 24);
          sum0            = madd256_ps(w0, delta_src, sum0);
          sum1            = madd256_ps(w1, delta_src, sum1);
          sum2            = madd256_ps(w2, delta_src, sum2);
          sum3 = madd128_ss(w3, _mm256_castps256_ps128(delta_src), sum3);
        }
      } else {
        for (size_t outc = 0; outc < out.depth_; ++outc, widx += wstep) {
          if (!tbl.is_connected(outc, inc)) {
            continue;
          }
          delta_src       = _mm256_broadcast_ss(&curr_delta[outc]);
          const float *pw = (const float *)&W[widx];
          __m256 w0       = _mm256_loadu_ps(pw + 0);
          __m256 w1       = _mm256_loadu_ps(pw + 8);
          __m256 w2       = _mm256_loadu_ps(pw + 16);
          __m128 w3       = _mm_load_ss(pw + 24);
          sum0            = madd256_ps(w0, delta_src, sum0);
          sum1            = madd256_ps(w1, delta_src, sum1);
          sum2            = madd256_ps(w2, delta_src, sum2);
          sum3 = madd128_ss(w3, _mm256_castps256_ps128(delta_src), sum3);
        }
      }
      float *delta_dst0 = pdelta_dst_org;
      float *delta_dst1 = &pdelta_dst_org[in_padded.width_ * 1];
      float *delta_dst2 = &pdelta_dst_org[in_padded.width_ * 2];
      float *delta_dst3 = &pdelta_dst_org[in_padded.width_ * 3];
      float *delta_dst4 = &pdelta_dst_org[in_padded.width_ * 4];
      __m256 dst0       = _mm256_loadu_ps(delta_dst0);
      __m256 dst1       = _mm256_loadu_ps(delta_dst1);
      __m256 dst2       = _mm256_loadu_ps(delta_dst2);
      __m256 dst3       = _mm256_loadu_ps(delta_dst3);
      __m256 dst4       = _mm256_maskload_ps(delta_dst4, imask);

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
      __m256 new_sum0 =
        _mm256_blend_ps(_mm256_setzero_ps(), sum0, 0x1F /* 0b00011111 */);
      __m256 new_sum1 =
        _mm256_blend_ps(_mm256_setzero_ps(),
                        _mm256_or_ps(rightShift<20>(sum0), leftShift<12>(sum1)),
                        0x1F /* 0b00011111 */);
      __m256 new_sum2 = _mm256_blend_ps(
        _mm256_setzero_ps(), rightShift<8>(sum1), 0x1F /* 0b00011111 */);
      __m256 new_sum3 =
        _mm256_blend_ps(_mm256_setzero_ps(),
                        _mm256_or_ps(rightShift<28>(sum1), leftShift<4>(sum2)),
                        0x1F /* 0b00011111 */);
      __m256 new_sum4 =
        _mm256_blend_ps(_mm256_setzero_ps(),
                        _mm256_set_m128(sum3, _mm256_extractf128_ps(sum2, 1)),
                        0x1F /* 0b00011111 */);
      dst0 = _mm256_add_ps(dst0, new_sum0);
      dst1 = _mm256_add_ps(dst1, new_sum1);
      dst2 = _mm256_add_ps(dst2, new_sum2);
      dst3 = _mm256_add_ps(dst3, new_sum3);
      dst4 = _mm256_add_ps(dst4, new_sum4);

      _mm256_maskstore_ps(delta_dst0, imask, dst0);
      _mm256_maskstore_ps(delta_dst1, imask, dst1);
      _mm256_maskstore_ps(delta_dst2, imask, dst2);
      _mm256_maskstore_ps(delta_dst3, imask, dst3);
      _mm256_maskstore_ps(delta_dst4, imask, dst4);
    }  // for
  } else {
    for (size_t inc = 0; inc < in.depth_;
         ++inc, pdelta_dst_org += in_padded_area) {
      for (size_t outc = 0; outc < out.depth_; ++outc) {
        if (!tbl.is_connected(outc, inc)) continue;

        const float *pw         = &W[25 * (in.depth_ * outc + inc)];
        const float *pdelta_src = &curr_delta[out.get_index(0, 0, outc)];
        float *pdelta_dst       = pdelta_dst_org;
        __m256 w0a              = _mm256_maskload_ps(pw + 0, imask);
        __m256 w1a              = _mm256_maskload_ps(pw + 5, imask);
        __m256 w2a              = _mm256_maskload_ps(pw + 10, imask);
        __m256 w3a              = _mm256_maskload_ps(pw + 15, imask);
        __m256 w4a              = _mm256_maskload_ps(pw + 20, imask);
        for (size_t y = 0; y < out_height;
             ++y, pdelta_src += out_width, pdelta_dst += h_stride2) {
          float *delta_dst0 = pdelta_dst;
          float *delta_dst1 = &pdelta_dst[in_padded.width_ * 1];
          float *delta_dst2 = &pdelta_dst[in_padded.width_ * 2];
          float *delta_dst3 = &pdelta_dst[in_padded.width_ * 3];
          float *delta_dst4 = &pdelta_dst[in_padded.width_ * 4];
          for (size_t x = 0; x < out_width; ++x) {
            __m256 delta_src = _mm256_broadcast_ss(pdelta_src + x);
            __m256 dst0      = _mm256_loadu_ps(delta_dst0);
            __m256 dst1      = _mm256_loadu_ps(delta_dst1);
            __m256 dst2      = _mm256_loadu_ps(delta_dst2);
            __m256 dst3      = _mm256_loadu_ps(delta_dst3);
            __m256 dst4      = _mm256_maskload_ps(delta_dst4, imask);
            dst0             = madd256_ps(w0a, delta_src, dst0);
            dst1             = madd256_ps(w1a, delta_src, dst1);
            dst2             = madd256_ps(w2a, delta_src, dst2);
            dst3             = madd256_ps(w3a, delta_src, dst3);
            dst4             = madd256_ps(w4a, delta_src, dst4);
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
          }  // for x
        }    // for y
      }      // for outc
    }        // for inc
  }

  accumulate_dw(params, prev_out, curr_delta, dW, db);

  if (params.has_bias) {
    accumulate_db(out, curr_delta, db);
  }
}  // avx_conv2d_5x5_back_kernel float ver

// double ver
template <typename Allocator>
void avx_conv2d_5x5_back_kernel(
  const core::conv_params &params,
  const std::vector<std::vector<double, Allocator>> &prev_out,
  const std::vector<double, Allocator> &W,
  std::vector<std::vector<double, Allocator>> &dW,
  std::vector<std::vector<double, Allocator>> &db,
  std::vector<std::vector<double, Allocator>> &curr_delta,
  std::vector<std::vector<double, Allocator>> &prev_delta,
  bool layer_parallelize) {
  // backward-pass fallbacks to tiny-backend when float_t is double
  conv2d_op_internal(prev_out, W, dW, db, curr_delta, prev_delta, params,
                     layer_parallelize);
}

// float ver
template <typename Allocator>
void avx_conv2d_5x5_back_kernel(
  const core::conv_params &params,
  const std::vector<std::vector<float, Allocator>> &prev_out,
  const std::vector<float, Allocator> &W,
  std::vector<std::vector<float, Allocator>> &dW,
  std::vector<std::vector<float, Allocator>> &db,
  std::vector<std::vector<float, Allocator>> &curr_delta,
  std::vector<std::vector<float, Allocator>> &prev_delta,
  bool layer_parallelize) {
  for_i(layer_parallelize, prev_out.size(), [&](size_t sample) {
    avx_conv2d_5x5_back_kernel_one(params, prev_out[sample], W, dW[sample],
                                   db[sample], curr_delta[sample],
                                   &prev_delta[sample]);
  });
}

#endif  // CNN_USE_AVX

inline void conv2d_grad_op_avx(const tensor_t &prev_out,
                               const vec_t &W,
                               tensor_t &dW,
                               tensor_t &db,
                               tensor_t &curr_delta,
                               tensor_t &prev_delta,
                               const core::conv_params &params,
                               const bool layer_parallelize) {
#ifdef CNN_USE_AVX
  if (params.weight.height_ == 5 && params.weight.width_ == 5) {
    avx_conv2d_5x5_back_kernel(params, prev_out, W, dW, db, curr_delta,
                               prev_delta, layer_parallelize);
    return;
  }
#endif

  conv2d_op_internal(prev_out, W, dW, db, curr_delta, prev_delta, params,
                     layer_parallelize);
}

}  // namespace kernels
}  // namespace tiny_dnn
