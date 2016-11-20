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

#include "tiny_dnn/core/params/deconv_params.h"

namespace tiny_dnn {
namespace core {
namespace kernels {

inline void tiny_deconv2d_back_kernel(const deconv_params& params,
                                      const tensor_t& prev_out,
                                      const vec_t& W,
                                      tensor_t&       dW,
                                      tensor_t&       db,
                                      tensor_t&       curr_delta,
                                      tensor_t*       prev_delta) {
    // propagate delta to previous layer
    for_i(prev_out.size(), [&](int sample) {
        for (serial_size_t inc = 0; inc < params.in.depth_; inc++) {
            for (serial_size_t outc = 0; outc < params.out.depth_; outc++) {
                if (!params.tbl.is_connected(outc, inc)) continue;

                serial_size_t idx = 0;
                idx = params.in.depth_ * outc + inc;
                idx = params.weight.get_index(0, 0, idx);
                const float_t *pw = &W[idx];

                idx = params.out_unpadded.get_index(0, 0, outc);
                const float_t *pdelta_src = &curr_delta[sample][idx];

                idx = params.in.get_index(0, 0, inc);
                float_t *pdelta_dst = &(*prev_delta)[sample][idx];

                for (serial_size_t y = 0; y < params.in.height_; y++) {
                    for (serial_size_t x = 0; x < params.in.width_; x++) {
                        const float_t * ppw = pw;

                        float_t * ppdelta_dst = pdelta_dst + y * params.in.width_ + x;
                        float_t sum = float_t(0);

                        for (serial_size_t wy = 0; wy < params.weight.height_; wy++) {
                            for (serial_size_t wx = 0; wx < params.weight.width_; wx++) {
                                idx = (y * params.h_stride + wy) *
                                    params.out.width_ + (x *
                                    params.w_stride + wx);
                                sum += ppw[wy * params.weight.width_ + wx] *
                                    pdelta_src[idx];
                            }
                        }
                        *ppdelta_dst += sum;
                    }
                }
            }
        }

        // accumulate dw
        for (serial_size_t inc = 0; inc < params.in.depth_; inc++) {
            for (serial_size_t outc = 0; outc < params.out.depth_; outc++) {
                if (!params.tbl.is_connected(outc, inc)) continue;

                for (serial_size_t wy = 0; wy < params.weight.height_; wy++) {
                    for (serial_size_t wx = 0; wx < params.weight.width_; wx++) {
                        float_t dst = float_t(0);

                        serial_size_t idx = 0;
                        idx = params.in.get_index(0, 0, inc);
                        const float_t * prevo = &prev_out[sample][idx];

                        idx = params.out.get_index(wx, wy, outc);
                        const float_t * delta = &curr_delta[sample][idx];

                        for (serial_size_t y = 0; y < params.in.height_; y++) {
                            dst += vectorize::dot(
                                prevo + y * params.in.width_,
                                delta + y * params.out.width_,
                                params.in.width_);
                        }

                        idx = params.in.depth_ * outc + inc;
                        dW[sample][params.weight.get_index(wx, wy, idx)] += dst;
                    }
                }
            }
        }

        // accumulate db
        if (params.has_bias) {
            //vec_t& db = *in_grad[2];

            for (serial_size_t outc = 0; outc < params.out.depth_; outc++) {
                serial_size_t idx = params.out.get_index(0, 0, outc);
                const float_t * delta = &curr_delta[sample][idx];
                const float_t * deltaa = delta + params.out.width_ *
                    params.out.height_;
                db[sample][outc] += std::accumulate(delta, deltaa, float_t(0));
            }
        }
    });
}

}  // namespace kernels
}  // namespace core
}  // namespace tiny_dnn
