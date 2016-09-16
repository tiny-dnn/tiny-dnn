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

#include "tiny_dnn/core/params/conv_params.h"

namespace tiny_dnn {

/* Helper class with tipical convolution routines
 */
class Conv2d {
 public:
    /* Applies padding to an input tensor given the convolution parameters
     *
     * @param in The input tensor
     * @param out The output tensor with padding applied
     */
    void copy_and_pad_input(const tensor_t& in, tensor_t& out) {
        if (params_.pad_type == core::padding::valid) {
            out = in;
            return;
        }

        tensor_t buf(in.size());

        for_i(true, buf.size(), [&](int sample) {
            // alloc temporary buffer.
            buf[sample].resize(params_.in.depth_ *
                               params_.in_padded.height_ *
                               params_.in_padded.width_);

            // make padded version in order to avoid corner-case in fprop/bprop
            for (cnn_size_t c = 0; c < params_.in.depth_; c++) {
                float_t* pimg = &buf[sample][params_.in_padded.get_index(
                                             params_.weight.width_  / 2,
                                             params_.weight.height_ / 2, c)];
                const float_t* pin = &in[sample][params_.in.get_index(0, 0, c)];

                for (cnn_size_t y = 0; y < params_.in.height_; y++) {
                    std::copy(pin, pin + params_.in.width_, pimg);
                    pin  += params_.in.width_;
                    pimg += params_.in_padded.width_;
                }
            }
        });

        // shrink buffer to output
        out = buf;
    }

    /* Applies unpadding to an input tensor given the convolution parameters
     *
     * @param in The input tensor
     * @param out The output tensor with unpadding applied
     */
    void copy_and_unpad_delta(const tensor_t& delta, tensor_t& delta_unpadded) {
        if (params_.pad_type == core::padding::valid) {
            delta_unpadded = delta;
            return;
        }
        
        tensor_t buf(delta.size());

        for_i(true, buf.size(), [&](int sample) {
            // alloc temporary buffer.
            buf[sample].resize(params_.in.depth_ *
                               params_.in.height_ *
                               params_.in.width_);

            for (cnn_size_t c = 0; c < params_.in.depth_; c++) {
                const float_t *pin =
                    &delta[sample][params_.in_padded.get_index(
                                   params_.weight.width_  / 2,
                                   params_.weight.height_ / 2, c)];
                float_t *pdst = &buf[sample][params_.in.get_index(0, 0, c)];

                for (cnn_size_t y = 0; y < params_.in.height_; y++) {
                    std::copy(pin, pin + params_.in.width_, pdst);
                    pdst += params_.in.width_;
                    pin  += params_.in_padded.width_;
                }
            }
        });

        // shrink buffer to output
        delta_unpadded = buf;
    }

    // Set and cast a params raw pointer to a specific convolution
    // operation parameters
    void setParams(core::Params* params) {
        params_ = cast_conv_params(params);
    }

    // Returns the convolution parameters
    core::conv_params params() const { return params_; }

 private:
    /* Cast a params raw pointer to a specific convolution operation parameters
     *
     * @param params The raw pointer to the parameters
     */
    core::conv_params& cast_conv_params(core::Params* params) const {
        return *(static_cast<core::conv_params*>(params));
    }

 private:
    /* The convolution parameters */
    core::conv_params params_;
};

}  // namespace tiny_dnn
