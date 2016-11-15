/*
    COPYRIGHT

    All contributions by Taiga Nomi
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.

    All other contributions:
    Copyright (c) 2013-2016, the respective contributors.
    All rights reserved.

    Each contributor holds copyright over their respective contributions.
    The project versioning (Git) records all such contribution source information.

    LICENSE

    The BSD 3-Clause License


    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name of tiny-dnn nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

namespace tiny_dnn {
namespace kernels {

inline void
conv2d_op_internal(const tensor_t&         in_data,
                   const vec_t&                  W,
                   const vec_t&               bias,
                   tensor_t&              out_data,
                   const core::conv_params& params,
                   const bool          parallelize) {
    for_i(parallelize, in_data.size(), [&](int sample) {
        const vec_t& in = in_data[sample];
        vec_t& a = out_data[sample];

        for (serial_size_t o = 0; o < params.out.depth_; o++) {
            for (serial_size_t inc = 0; inc < params.in.depth_; inc++) {
                if (!params.tbl.is_connected(o, inc)) continue;

                serial_size_t idx = 0;
                idx = params.in.depth_ * o + inc;
                idx = params.weight.get_index(0, 0, idx);
                const float_t *pw = &W[idx];

                idx = params.in_padded.get_index(0, 0, inc);
                const float_t *pi = &in[idx];

                idx = params.out.get_index(0, 0, o);
                float_t *pa = &a[idx];

                for (serial_size_t y = 0; y < params.out.height_; y++) {
                    for (serial_size_t x = 0; x < params.out.width_; x++) {
                        const float_t * ppw = pw;
                        const float_t * ppi = pi + params.in_padded.width_ *
                            (y * params.h_stride) +
                            x * params.w_stride;
                        float_t sum = float_t(0);

                        // should be optimized for small kernel(3x3,5x5)
                        for (serial_size_t wy = 0; wy < params.weight.height_; wy++) {    // NOLINT
                            for (serial_size_t wx = 0; wx < params.weight.width_; wx++) { // NOLINT
                                idx = wy * params.in_padded.width_ + wx;
                                sum += *ppw++ * ppi[idx];
                            }
                        }
                        pa[y * params.out.width_ + x] += sum;
                    }
                }
            }

            if (params.has_bias) {
                float_t * pa = &a[params.out.get_index(0, 0, o)];
                float_t * paa = pa + params.out.width_ * params.out.height_;
                std::for_each(pa, paa, [&](float_t& f) { f += bias[o]; });
            }
        }
    });
}


/******************************************************************/


template <typename tensor_t, typename vec_t>
void
conv2d_op_internal(const tensor_t&        prev_out,
                   const vec_t&                  W,
                   tensor_t&                    dW,
                   tensor_t&                    db,
                   tensor_t&            curr_delta,
                   tensor_t&            prev_delta,
                   const core::conv_params& params,
                   const bool          parallelize) {

    typedef typename vec_t::value_type float_t;

    for_i(parallelize, prev_out.size(), [&](int sample) {
        // propagate delta to previous layer
        for (serial_size_t inc = 0; inc < params.in.depth_; inc++) {
            for (serial_size_t outc = 0; outc < params.out.depth_; outc++) {
                if (!params.tbl.is_connected(outc, inc)) continue;

                serial_size_t idx = 0;
                idx = params.in.depth_ * outc + inc;
                idx = params.weight.get_index(0, 0, idx);
                const float_t *pw = &W[idx];

                idx = params.out.get_index(0, 0, outc);
                const float_t *pdelta_src = &curr_delta[sample][idx];

                idx = params.in_padded.get_index(0, 0, inc);
                //float_t *pdelta_dst = &(*prev_delta)[sample][idx];
                float_t *pdelta_dst = &prev_delta[sample][idx];

                for (serial_size_t y = 0; y < params.out.height_; y++) {
                    for (serial_size_t x = 0; x < params.out.width_; x++) {
                        const float_t * ppw = pw;

                        idx = y * params.out.width_ + x;
                        const float_t ppdelta_src = pdelta_src[idx];

                        float_t * ppdelta_dst = pdelta_dst +
                                y * params.h_stride * params.in_padded.width_ +
                                x * params.w_stride;

                        for (serial_size_t wy = 0; wy < params.weight.height_; wy++) {    // NOLINT
                            for (serial_size_t wx = 0; wx < params.weight.width_; wx++) { // NOLINT
                                idx = wy * params.in_padded.width_ + wx;
                                ppdelta_dst[idx] += *ppw++ * ppdelta_src;
                            }
                        }
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
                        idx = params.in_padded.get_index(wx, wy, inc);
                        const float_t * prevo = &prev_out[sample][idx];

                        idx = params.out.get_index(0, 0, outc);
                        const float_t * delta = &curr_delta[sample][idx];

                        if (params.w_stride > 1) {
                            for (serial_size_t y = 0; y < params.out.height_; y++) {
                                serial_size_t prevo_idx = y * params.in_padded.width_ * params.h_stride;
                                serial_size_t delta_idx = y * params.out.width_;

                                for (serial_size_t x = 0; x < params.out.width_; x++) {
                                    dst += prevo[prevo_idx + x * params.w_stride] * delta[delta_idx + x];
                                }
                            }
                        } else {
                            for (serial_size_t y = 0; y < params.out.height_; y++) {
                                dst += vectorize::dot(
                                    prevo + y * params.in_padded.width_ * params.h_stride,
                                    delta + y * params.out.width_,
                                    params.out.width_);
                            }
                        }


                        idx = params.in.depth_ * outc + inc;
                        dW[sample][params.weight.get_index(wx, wy, idx)] += dst;
                    }
                }
            }
        }

        // accumulate db
        if (params.has_bias) {
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
}  // namespace tiny_dnn
