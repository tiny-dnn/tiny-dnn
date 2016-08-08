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

    * Neither the name of tiny-cnn nor the names of its
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

#include "tiny_cnn/core/kernels/conv2d.h"
#include "tiny_cnn/core/framework/op_kernel.h"

namespace tiny_cnn {

class Conv2dCustomForwardOp : public core::OpKernel, Conv2d {
 public:
    explicit Conv2dCustomForwardOp(core::OpKernelConstruction* context)
        : core::OpKernel(context) {}

    void compute(core::OpKernelContext* context) override {
        const tensor_t& in_data = context->input(0);
        const vec_t&          W = context->input(1)[0];
        const vec_t&       bias = context->input(2)[0];
        tensor_t&      out_data = context->output(1);

        // retrieve the convolutional parameters and pad input
        Conv2d::setParams(context->params());

        // pad input data
        tensor_t in_data_padded;
        Conv2d::copy_and_pad_input(in_data, in_data_padded);

        core::conv_params params = Conv2d::params();
        bool layer_parallelize = context->parallelize();

        // convolution algorithm

        for_i(layer_parallelize, in_data_padded.size(), [&](int sample) {
            const vec_t& in = in_data_padded[sample];
            vec_t& a = out_data[sample];

            for (cnn_size_t o = 0; o < params.out.depth_; o++) {
                for (cnn_size_t inc = 0; inc < params.in.depth_; inc++) {
                    if (!params.tbl.is_connected(o, inc)) continue;

                    cnn_size_t idx = 0;
                    idx = params.in.depth_ * o + inc;
                    idx = params.weight.get_index(0, 0, idx);
                    const float_t *pw = &W[idx];

                    idx = params.in_padded.get_index(0, 0, inc);
                    const float_t *pi = &in[idx];

                    idx = params.out.get_index(0, 0, o);
                    float_t *pa = &a[idx];

                    for (cnn_size_t y = 0; y < params.out.height_; y++) {
                        for (cnn_size_t x = 0; x < params.out.width_; x++) {
                            const float_t * ppw = pw;
                            const float_t * ppi = pi + params.in_padded.width_ *
                                (y * params.h_stride) +
                                x * params.w_stride;
                            float_t sum = float_t(0);

                            // should be optimized for small kernel(3x3,5x5)
                            for (cnn_size_t wy = 0; wy < params.weight.height_; wy++) {    // NOLINT
                                for (cnn_size_t wx = 0; wx < params.weight.width_; wx++) { // NOLINT
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
};

class Conv2dCustomBackwardOp : public core::OpKernel, Conv2d {
 public:
    explicit Conv2dCustomBackwardOp(core::OpKernelConstruction* context)
        : core::OpKernel(context) {}

    void compute(core::OpKernelContext* context) override {

        //std::vector<const vec_t*>& prev_out = cws.prev_out_padded_;
        const tensor_t& prev_out = context->input(0);

        const vec_t& W  = context->input(1)[0];
        tensor_t&    dW = context->input_grad(1);
        tensor_t&    db = context->input_grad(2);
        tensor_t&    curr_delta = context->output_grad(1);
 
        //tensor_t*    prev_delta = (params_c_->pad_type == padding::same) ?
        //                           &cws.prev_delta_padded_ : in_grad[0];
        tensor_t&    prev_delta = context->input_grad(0);

        // retrieve the convolutional parameters
        Conv2d::setParams(context->params());
        core::conv_params params = Conv2d::params();
        bool layer_parallelize = context->parallelize();

        // fill_tensor(prev_delta, float_t(0));
        //kernels::tiny_conv2d_back_kernel(*params_c_,
        //    prev_out, W, dW, db, curr_delta, prev_delta);

        // propagate delta to previous layer

        for_i(prev_out.size(), [&](int sample) {
            for (cnn_size_t inc = 0; inc < params.in.depth_; inc++) {
                for (cnn_size_t outc = 0; outc < params.out.depth_; outc++) {
                    if (!params.tbl.is_connected(outc, inc)) continue;

                    cnn_size_t idx = 0;
                    idx = params.in.depth_ * outc + inc;
                    idx = params.weight.get_index(0, 0, idx);
                    const float_t *pw = &W[idx];

                    idx = params.out.get_index(0, 0, outc);
                    const float_t *pdelta_src = &curr_delta[sample][idx];

                    idx = params.in_padded.get_index(0, 0, inc);
                    //float_t *pdelta_dst = &(*prev_delta)[sample][idx];
                    float_t *pdelta_dst = &prev_delta[sample][idx];

                    for (cnn_size_t y = 0; y < params.out.height_; y++) {
                        for (cnn_size_t x = 0; x < params.out.width_; x++) {
                            const float_t * ppw = pw;

                            idx = y * params.out.width_ + x;
                            const float_t ppdelta_src = pdelta_src[idx];

                            float_t * ppdelta_dst = pdelta_dst +
                                  y * params.h_stride * params.in_padded.width_ +
                                  x * params.w_stride;

                            for (cnn_size_t wy = 0; wy < params.weight.height_; wy++) {    // NOLINT
                                for (cnn_size_t wx = 0; wx < params.weight.width_; wx++) { // NOLINT
                                    idx = wy * params.in_padded.width_ + wx;
                                    ppdelta_dst[idx] += *ppw++ * ppdelta_src;
                                }
                            }
                        }
                    }
                }
            }

            // accumulate dw
            for (cnn_size_t inc = 0; inc < params.in.depth_; inc++) {
                for (cnn_size_t outc = 0; outc < params.out.depth_; outc++) {
                    if (!params.tbl.is_connected(outc, inc)) continue;

                    for (cnn_size_t wy = 0; wy < params.weight.height_; wy++) {
                        for (cnn_size_t wx = 0; wx < params.weight.width_; wx++) {
                            float_t dst = float_t(0);

                            cnn_size_t idx = 0;
                            idx = params.in_padded.get_index(wx, wy, inc);
                            //const float_t * prevo = &(*prev_out[sample])[idx];
                            const float_t * prevo = &prev_out[sample][idx];

                            idx = params.out.get_index(0, 0, outc);
                            const float_t * delta = &curr_delta[sample][idx];

                            for (cnn_size_t y = 0; y < params.out.height_; y++) {
                                dst += vectorize::dot(
                                    prevo + y * params.in_padded.width_,
                                    delta + y * params.out.width_,
                                    params.out.width_);
                            }

                            idx = params.in.depth_ * outc + inc;
                            dW[sample][params.weight.get_index(wx, wy, idx)] += dst;
                        }
                    }
                }
            }

            // accumulate db
            if (params.has_bias) {
                for (cnn_size_t outc = 0; outc < params.out.depth_; outc++) {
                    cnn_size_t idx = params.out.get_index(0, 0, outc);
                    const float_t * delta = &curr_delta[sample][idx];
                    const float_t * deltaa = delta + params.out.width_ *
                        params.out.height_;
                    db[sample][outc] += std::accumulate(delta, deltaa, float_t(0));
                }
            }
        });

        // apply unpadding
        if (params.pad_type == core::padding::same) {
            copy_and_unpad_delta(prev_out, prev_delta);
        }
    }
};

}  // namespace tiny_cnn
