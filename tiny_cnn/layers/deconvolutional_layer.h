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
#include "tiny_cnn/util/util.h"
#include "tiny_cnn/util/image.h"
#include "tiny_cnn/activations/activation_function.h"
#include <deque>

namespace tiny_cnn {

/**
 * 2D deconvolution layer
 *
 * take input as two-dimensional *image* and applying filtering operation.
 **/
template<typename Activation = activation::identity>
class deconvolutional_layer : public feedforward_layer<Activation> {
public:
    typedef feedforward_layer<Activation> Base;
    CNN_USE_LAYER_MEMBERS;

    /**
    * constructing deconvolutional layer
    *
    * @param in_width     [in] input image width
    * @param in_height    [in] input image height
    * @param window_size  [in] window(kernel) size of convolution
    * @param in_channels  [in] input image channels (grayscale=1, rgb=3)
    * @param out_channels [in] output image channels
    * @param padding      [in] rounding strategy
    *                          valid: use valid pixels of input only. output-size = (in-width - window_size + 1) * (in-height - window_size + 1) * out_channels
    *                          same: add zero-padding to keep same width/height. output-size = in-width * in-height * out_channels
    * @param has_bias     [in] whether to add a bias vector to the filter outputs
    * @param w_stride     [in] specify the horizontal interval at which to apply the filters to the input
    * @param h_stride     [in] specify the vertical interval at which to apply the filters to the input
    **/
    deconvolutional_layer(cnn_size_t in_width,
                        cnn_size_t in_height,
                        cnn_size_t window_size,
                        cnn_size_t in_channels,
                        cnn_size_t out_channels,
                        padding    pad_type = padding::valid,
                        bool       has_bias = true,
                        cnn_size_t w_stride = 1,
                        cnn_size_t h_stride = 1)
        : Base(std_input_order(has_bias))
    {
        deconv_set_params(shape3d(in_width, in_height, in_channels), window_size, window_size,
                        out_channels, pad_type, has_bias, w_stride, h_stride);
    }

    /**
    * constructing convolutional layer
    *
    * @param in_width      [in] input image width
    * @param in_height     [in] input image height
    * @param window_width  [in] window_width(kernel) size of convolution
    * @param window_height [in] window_height(kernel) size of convolution
    * @param in_channels   [in] input image channels (grayscale=1, rgb=3)
    * @param out_channels  [in] output image channels
    * @param padding       [in] rounding strategy
    *                          valid: use valid pixels of input only. output-size = (in-width - window_width + 1) * (in-height - window_height + 1) * out_channels
    *                          same: add zero-padding to keep same width/height. output-size = in-width * in-height * out_channels
    * @param has_bias     [in] whether to add a bias vector to the filter outputs
    * @param w_stride     [in] specify the horizontal interval at which to apply the filters to the input
    * @param h_stride     [in] specify the vertical interval at which to apply the filters to the input
    **/
    deconvolutional_layer(cnn_size_t in_width,
                        cnn_size_t in_height,
                        cnn_size_t window_width,
                        cnn_size_t window_height,
                        cnn_size_t in_channels,
                        cnn_size_t out_channels,
                        padding    pad_type = padding::valid,
                        bool       has_bias = true,
                        cnn_size_t w_stride = 1,
                        cnn_size_t h_stride = 1)
        : Base(std_input_order(has_bias))
    {
        deconv_set_params(shape3d(in_width, in_height, in_channels), window_width, window_height,
                        out_channels, pad_type, has_bias, w_stride, h_stride);
    }

    /**
    * constructing convolutional layer
    *
    * @param in_width         [in] input image width
    * @param in_height        [in] input image height
    * @param window_size      [in] window(kernel) size of convolution
    * @param in_channels      [in] input image channels (grayscale=1, rgb=3)
    * @param out_channels     [in] output image channels
    * @param connection_table [in] definition of connections between in-channels and out-channels
    * @param pad_type         [in] rounding strategy
    *                               valid: use valid pixels of input only. output-size = (in-width - window_size + 1) * (in-height - window_size + 1) * out_channels
    *                               same: add zero-padding to keep same width/height. output-size = in-width * in-height * out_channels
    * @param has_bias         [in] whether to add a bias vector to the filter outputs
    * @param w_stride         [in] specify the horizontal interval at which to apply the filters to the input
    * @param h_stride         [in] specify the vertical interval at which to apply the filters to the input
    **/
    deconvolutional_layer(cnn_size_t              in_width,
                        cnn_size_t              in_height,
                        cnn_size_t              window_size,
                        cnn_size_t              in_channels,
                        cnn_size_t              out_channels,
                        const connection_table& connection_table,
                        padding                 pad_type = padding::valid,
                        bool                    has_bias = true,
                        cnn_size_t              w_stride = 1,
                        cnn_size_t              h_stride = 1)
        : Base(std_input_order(has_bias)), tbl_(connection_table)
    {
        deconv_set_params(shape3d(in_width, in_height, in_channels), window_size, window_size,
                        out_channels, pad_type, has_bias, w_stride, h_stride);
    }

    /**
    * constructing convolutional layer
    *
    * @param in_width         [in] input image width
    * @param in_height        [in] input image height
    * @param window_width     [in] window_width(kernel) size of convolution
    * @param window_height    [in] window_height(kernel) size of convolution
    * @param in_channels      [in] input image channels (grayscale=1, rgb=3)
    * @param out_channels     [in] output image channels
    * @param connection_table [in] definition of connections between in-channels and out-channels
    * @param pad_type         [in] rounding strategy
    *                               valid: use valid pixels of input only. output-size = (in-width - window_size + 1) * (in-height - window_size + 1) * out_channels
    *                               same: add zero-padding to keep same width/height. output-size = in-width * in-height * out_channels
    * @param has_bias         [in] whether to add a bias vector to the filter outputs
    * @param w_stride         [in] specify the horizontal interval at which to apply the filters to the input
    * @param h_stride         [in] specify the vertical interval at which to apply the filters to the input
    **/
    deconvolutional_layer(cnn_size_t              in_width,
                        cnn_size_t              in_height,
                        cnn_size_t              window_width,
                        cnn_size_t              window_height,
                        cnn_size_t              in_channels,
                        cnn_size_t              out_channels,
                        const connection_table& connection_table,
                        padding                 pad_type = padding::valid,
                        bool                    has_bias = true,
                        cnn_size_t              w_stride = 1,
                        cnn_size_t              h_stride = 1)
        : Base(has_bias ? 3 : 2, 1, std_input_order(has_bias)), tbl_(connection_table)
    {
        deconv_set_params(shape3d(in_width, in_height, in_channels), window_width, window_height,
                        out_channels, pad_type, has_bias, w_stride, h_stride);
    }

    ///< number of incoming connections for each output unit
    virtual size_t fan_in_size() const override
    {
        return weight_.width_ * weight_.height_ * in_.depth_;
    }

    ///< number of outgoing connections for each input unit
    virtual size_t fan_out_size() const override
    {
        return (weight_.width_ * w_stride_) * (weight_.height_ * h_stride_) * out_.depth_;
    }

    void forward_propagation(cnn_size_t index,
                             const std::vector<vec_t*>& in_data,
                             std::vector<vec_t*>& out_data) override {
        deconv_layer_worker_storage_[index].prev_out_ = in_data[0];
        const vec_t& W   = *in_data[1];
        vec_t&       out = *out_data[0];
        vec_t&       a   = *out_data[1];
        const vec_t &in  = *(deconv_layer_worker_storage_[index].prev_out_); // input
        
        std::fill(a.begin(), a.end(), float_t(0));

        for_i(parallelize_, out_.depth_, [&](int o) {
            for (cnn_size_t inc = 0; inc < in_.depth_; inc++) {
                if (!tbl_.is_connected(o, inc)) continue;

                const float_t *pw = &W[weight_.get_index(0, 0, in_.depth_ * o + inc)];
                const float_t *pi = &in[in_.get_index(0, 0, inc)];
                float_t *pa = &a[out_.get_index(0, 0, o)];

                for (cnn_size_t y = 0; y < out_.height_; y++) {
                    for (cnn_size_t x = 0; x < out_.width_; x++) {
                        const float_t * ppw = pw;
                        const float_t * ppi = pi + (y * h_stride_) * in_.width_ + x * w_stride_;
                        // float_t sum = float_t(0);

                        // should be optimized for small kernel(3x3,5x5)
                        for (cnn_size_t wy = 0; wy < weight_.height_; wy++) {
                            for (cnn_size_t wx = 0; wx < weight_.width_; wx++) {
                                // sum += *ppw++ * ppi[(weight_.height_-1-wy) * in_.width_ + (weight_.width_-1-wx)];
                                pa[(y * out_.width_ + x) + (wy * weight_.width_ + wx)] += *ppw++ * ppi[y * in_.width_ + x];
                            }
                        }
                        // pa[y * out_.width_ + x] += sum;
                    }
                }
            }

            if (has_bias_) {
                const vec_t& bias = *in_data[2];
                float_t *pa = &a[out_.get_index(0, 0, o)];
                float_t b = bias[o];
                std::for_each(pa, pa + out_.width_ * out_.height_, [&](float_t& f) { f += b; });
            }
        });

        for_i(parallelize_, out_.size(), [&](int i) {
            out[i] = h_.f(a, i);
        });
        copy_and_unpad_output(out, static_cast<int>(index));
    }

    float_t& weight_at(cnn_size_t in_channel, cnn_size_t out_channel, cnn_size_t kernel_x, cnn_size_t kernel_y) {
        vec_t* W = this->get_weights()[0];
        return W[weight_.get_index(kernel_x, kernel_y, in_.depth_ * out_channel + in_channel)];
    }

    void back_propagation(cnn_size_t                 index,
                          const std::vector<vec_t*>& in_data,
                          const std::vector<vec_t*>& out_data,
                          std::vector<vec_t*>&       out_grad,
                          std::vector<vec_t*>&       in_grad) override {

        deconv_layer_worker_specific_storage& cws = deconv_layer_worker_storage_[index];

        const vec_t& prev_out = *(cws.cur_out_padded_);
        const vec_t& W = *in_data[1];
        vec_t*       prev_delta = (pad_type_ == padding::same) ? &cws.prev_delta_padded_ : in_grad[0];
        vec_t&       dW = *in_grad[1];
        vec_t&       curr_delta = *out_grad[1];

        assert(W.size() == weight_.size());
        assert(dW.size() == weight_.size());
        assert(curr_delta.size() == out_shape()[0].size());

        this->backward_activation(*out_grad[0], *out_data[0], curr_delta);

        std::fill(prev_delta->begin(), prev_delta->end(), float_t(0));

        // propagate delta to previous layer
        for_i(in_.depth_, [&](int inc) {
            for (cnn_size_t outc = 0; outc < out_.depth_; outc++) {
                if (!tbl_.is_connected(outc, inc)) continue;

                const float_t *pw = &W[weight_.get_index(0, 0, in_.depth_ * outc + inc)];
                const float_t *pdelta_src = &curr_delta[out_.get_index(0, 0, outc)];
                float_t *pdelta_dst = &(*prev_delta)[in_.get_index(0, 0, inc)];

                for (cnn_size_t y = 0; y < out_.height_; y++) {
                    for (cnn_size_t x = 0; x < out_.width_; x++) {
                        const float_t * ppw = pw;
                        const float_t ppdelta_src = pdelta_src[y * out_.width_ + x];
                        float_t * ppdelta_dst = pdelta_dst + y * h_stride_ * in_.width_ + x * w_stride_;

                        for (cnn_size_t wy = 0; wy < weight_.height_; wy++) {
                            for (cnn_size_t wx = 0; wx < weight_.width_; wx++) {
                                ppdelta_dst[(weight_.height_-1-wy) * in_.width_ + (weight_.width_-1-wx)] += *ppw++ * ppdelta_src;
                            }
                        }
                    }
                }
            }
        });

        // accumulate dw
        for_i(in_.depth_, [&](int inc) {
            for (cnn_size_t outc = 0; outc < out_.depth_; outc++) {

                if (!tbl_.is_connected(outc, inc)) continue;

                for (cnn_size_t wy = 0; wy < weight_.height_; wy++) {
                    for (cnn_size_t wx = 0; wx < weight_.width_; wx++) {
                        float_t dst = float_t(0);
                        const float_t * prevo = &prev_out[in_.get_index(wx, wy, inc)];
                        const float_t * delta = &curr_delta[out_.get_index(0, 0, outc)];

                        for (cnn_size_t y = 0; y < out_.height_; y++) {
                            dst += vectorize::dot(prevo + y * in_.width_, delta + y * out_.width_, out_.width_);
                        }
                        dW[weight_.get_index(weight_.width_-wx-1, weight_.height_-wy-1, in_.depth_ * outc + inc)] += dst;
                    }
                }
            }
        });

        // accumulate db
        if (has_bias_) {
            vec_t& db = *in_grad[2];

            for (cnn_size_t outc = 0; outc < out_.depth_; outc++) {
                const float_t *delta = &curr_delta[out_.get_index(0, 0, outc)];
                db[outc] += std::accumulate(delta, delta + out_.width_ * out_.height_, float_t(0));
            }
        }

        if (pad_type_ == padding::same)
            copy_and_pad_delta(cws.prev_delta_padded_, *in_grad[0]);
    }

    std::vector<index3d<cnn_size_t>> in_shape() const override {
        if (has_bias_) {
            return{ in_, weight_, index3d<cnn_size_t>(1, 1, out_.depth_) };
        } else {
            return { in_, weight_ };
        }
    }

    std::vector<index3d<cnn_size_t>> out_shape() const override { return {out_, out_}; }
    std::string layer_type() const override { return "deconv"; }

    image<> weight_to_image() const {
        image<> img;
        const cnn_size_t border_width = 1;
        const auto pitch = weight_.width_ + border_width;
        const auto width = out_.depth_ * pitch + border_width;
        const auto height = in_.depth_ * pitch + border_width;
        const image<>::intensity_t bg_color = 255;
        const vec_t& W = *this->get_weights()[0];

        img.resize(width, height);
        img.fill(bg_color);

        auto minmax = std::minmax_element(W.begin(), W.end());

        for (cnn_size_t r = 0; r < in_.depth_; ++r) {
            for (cnn_size_t c = 0; c < out_.depth_; ++c) {
                if (!tbl_.is_connected(c, r)) continue;

                const auto top = r * pitch + border_width;
                const auto left = c * pitch + border_width;

                for (cnn_size_t y = 0; y < weight_.height_; ++y) {
                    for (cnn_size_t x = 0; x < weight_.width_; ++x) {
                        const float_t w = W[weight_.get_index(x, y, c * in_.depth_ + r)];

                        img.at(left + x, top + y)
                            = static_cast<image<>::intensity_t>(rescale(w, *minmax.first, *minmax.second, 0, 255));
                    }
                }
            }
        }
        return img;
    }

    virtual void set_worker_count(cnn_size_t worker_count) override {
        Base::set_worker_count(worker_count);
        deconv_layer_worker_storage_.resize(worker_count);
        init();
    }

private:
    void deconv_set_params(const shape3d& in,
                         cnn_size_t     w_width,
                         cnn_size_t     w_height,
                         cnn_size_t     outc,
                         padding        ptype,
                         bool           has_bias,
                         cnn_size_t     w_stride,
                         cnn_size_t     h_stride) {
        in_ = in;
        out_ = shape3d(deconv_out_length(in.width_, w_width, w_stride),
                       deconv_out_length(in.height_, w_height, h_stride),
                       outc);
        out_unpadded_ = shape3d(deconv_out_unpadded_length(in.width_, w_width, w_stride, ptype),
                             deconv_out_unpadded_length(in.height_, w_height, h_stride, ptype),
                             outc);
        weight_ = shape3d(w_width, w_height, in.depth_ * outc);
        has_bias_ = has_bias;
        pad_type_ = ptype;
        w_stride_ = w_stride;
        h_stride_ = h_stride;
    }

    void init() {
        for (deconv_layer_worker_specific_storage& cws : deconv_layer_worker_storage_) {
            if (pad_type_ == padding::same) {
                cws.prev_out_buf_.resize(in_.size(), float_t(0));
                cws.prev_delta_padded_.resize(in_.size(), float_t(0));
            }
            else {
                cws.prev_out_buf_.clear();
            }
        }
        if (pad_type_ == padding::same) {
            prev_delta2_padded_.resize(in_.size(), float_t(0));
        }
    }

    cnn_size_t in_length(cnn_size_t in_length, cnn_size_t window_size, padding pad_type) const {
        return pad_type == padding::same ? in_length : in_length;
    }

    static cnn_size_t deconv_out_length(cnn_size_t in_length, cnn_size_t window_size, cnn_size_t stride) {
        return (cnn_size_t)ceil((double)(in_length) * stride + window_size - 1);
    }

    static cnn_size_t deconv_out_unpadded_length(cnn_size_t in_length, cnn_size_t window_size, cnn_size_t stride, padding pad_type) {
        return pad_type == padding::same ? (cnn_size_t)ceil((double)in_length * stride) : (cnn_size_t)ceil((double)(in_length) * stride + window_size - 1);
    }

    static cnn_size_t deconv_out_dim(cnn_size_t in_width, cnn_size_t in_height, cnn_size_t window_size, cnn_size_t w_stride, cnn_size_t h_stride, padding pad_type) {
        return deconv_out_unpadded_length(in_width, window_size, w_stride, pad_type) * deconv_out_unpadded_length(in_height, window_size, h_stride, pad_type);
    }

    cnn_size_t deconv_out_dim(cnn_size_t in_width, cnn_size_t in_height, cnn_size_t window_width, cnn_size_t window_height, cnn_size_t w_stride, cnn_size_t h_stride, padding pad_type) const {
        return deconv_out_unpadded_length(in_width, window_width, w_stride, pad_type) * deconv_out_unpadded_length(in_height, window_height, h_stride, pad_type);
    }

    void copy_and_pad_delta(const vec_t& delta, vec_t& dst) {
        if (pad_type_ == padding::valid) {
            dst = delta;
        }
        else {
            for (cnn_size_t c = 0; c < in_.depth_; c++) {
                float_t *pdst = &dst[in_.get_index(0, 0, c)];
                const float_t *pin = &delta[in_.get_index(weight_.width_ / 2, weight_.height_ / 2, c)];

                for (cnn_size_t y = 0; y < in_.height_; y++, pdst += in_.width_, pin += in_.width_) {
                    std::copy(pin, pin + in_.width_, pdst);
                }
            }
        }
    }

    void copy_and_unpad_output(const vec_t& out, int worker_index) {
        deconv_layer_worker_specific_storage& cws = deconv_layer_worker_storage_[worker_index];

        vec_t* dst = &cws.cur_out_buf_;

        if (pad_type_ == padding::valid) {
            cws.cur_out_padded_ = &out;
        }
        else {
            // make unpadded version in order to restore scale in fprop/bprop
            for (cnn_size_t c = 0; c < out_.depth_; c++) {
                float_t *pimg = &(*dst)[out_unpadded_.get_index(0, 0, c)];
                const float_t *pout = &out[out_.get_index(weight_.width_ / 2, weight_.height_ / 2, c)];

                for (cnn_size_t y = weight_.height_ / 2; y < in_.height_ - weight_.height_ / 2; y++, pout += out_.width_, pimg += (out_.width_ - weight_.width_ + 1)) {
                    std::copy(pout, pout + out_.width_ - weight_.width_ + 1, pimg);
                }
            }
            cws.cur_out_padded_ = &cws.cur_out_buf_;
        }
    }

    struct deconv_layer_worker_specific_storage {
        const vec_t* prev_out_;
        const vec_t* cur_out_padded_;
        vec_t prev_out_buf_;
        vec_t cur_out_buf_;
        vec_t prev_delta_padded_;
    };

    std::vector<deconv_layer_worker_specific_storage> deconv_layer_worker_storage_;

    vec_t  prev_delta2_padded_;

    connection_table tbl_;
    index3d<cnn_size_t> in_;
    index3d<cnn_size_t> in_padded_;
    index3d<cnn_size_t> out_;
    index3d<cnn_size_t> out_unpadded_;
    index3d<cnn_size_t> weight_;
    bool has_bias_;
    padding pad_type_;
    size_t w_stride_;
    size_t h_stride_;
};

} // namespace tiny_cnn
