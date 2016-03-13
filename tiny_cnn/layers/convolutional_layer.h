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

struct connection_table {
    connection_table() : rows_(0), cols_(0) {}
    connection_table(const bool *ar, cnn_size_t rows, cnn_size_t cols) : connected_(rows * cols), rows_(rows), cols_(cols) {
        std::copy(ar, ar + rows * cols, connected_.begin());
    }
    connection_table(cnn_size_t ngroups, cnn_size_t rows, cnn_size_t cols) : connected_(rows * cols, false), rows_(rows), cols_(cols) {
        if (rows % ngroups || cols % ngroups) throw nn_error("invalid group size");

        cnn_size_t row_group = rows / ngroups;
        cnn_size_t col_group = cols / ngroups;

        for (cnn_size_t g = 0; g < ngroups; g++) {
            for (cnn_size_t r = 0; r < row_group; r++)
              for (cnn_size_t c = 0; c < col_group; c++)
                connected_[(r + g * row_group) * cols_ + c + g * col_group] = true;
        }
    }

    bool is_connected(cnn_size_t x, cnn_size_t y) const {
        return is_empty() ? true : connected_[y * cols_ + x];
    }

    bool is_empty() const {
        return rows_ == 0 && cols_ == 0;
    }

    std::deque<bool> connected_;
    cnn_size_t rows_;
    cnn_size_t cols_;
};

enum class padding {
    valid, ///< use valid pixels of input
    same   ///< add zero-padding around input so as to keep image size
};


template<typename Activation = activation::identity>
class convolutional_layer : public layer<Activation> {
public:
    typedef layer<Activation> Base;
    CNN_USE_LAYER_MEMBERS;

    using layer_base::out_size;

    /**
    * constructing convolutional layer
    *
    * @param in_width     [in] input image width
    * @param in_height    [in] input image height
    * @param window_size  [in] window(kernel) size of convolution
    * @param in_channels  [in] input image channels (grayscale=1, rgb=3)
    * @param out_channels [in] output image channels
    * @param padding      [in] rounding strategy
    *                          valid: use valid pixels of input only. output-size = (in-width - window_size + 1) * (in-height - window_size + 1) * out_channels
    *                          same: add zero-padding to keep same width/height. output-size = in-width * in-height * out_channels
    **/
    convolutional_layer(cnn_size_t in_width,
        cnn_size_t in_height,
        cnn_size_t window_size,
        cnn_size_t in_channels,
        cnn_size_t out_channels,
        padding pad_type = padding::valid,
        bool has_bias = true,
        cnn_size_t w_stride = 1,
        cnn_size_t h_stride = 1)
        : Base(in_width * in_height * in_channels, conv_out_dim(in_width, in_height, window_size, w_stride, h_stride, pad_type) * out_channels,
            sqr(window_size) * in_channels * out_channels, has_bias ? out_channels : 0),
        in_(in_width, in_height, in_channels),
        in_padded_(in_length(in_width, window_size, pad_type), in_length(in_height, window_size, pad_type), in_channels),
        out_(conv_out_length(in_width, window_size, w_stride, pad_type), conv_out_length(in_height, window_size, h_stride, pad_type), out_channels),
        weight_(window_size, window_size, in_channels*out_channels),
        pad_type_(pad_type),
        w_stride_(w_stride), h_stride_(h_stride)
    {
        init();
    }

    /**
    * constructing convolutional layer
    *
    * @param in_width         [in] input image width
    * @param in_height        [in] input image height
    * @param window_width  [in] window_width(kernel) size of convolution
    * @param window_height [in] window_height(kernel) size of convolution
    * @param in_channels   [in] input image channels (grayscale=1, rgb=3)
    * @param out_channels  [in] output image channels
    * @param padding       [in] rounding strategy
    *                          valid: use valid pixels of input only. output-size = (in-width - window_width + 1) * (in-height - window_height + 1) * out_channels
    *                          same: add zero-padding to keep same width/height. output-size = in-width * in-height * out_channels
    **/
    convolutional_layer(cnn_size_t in_width,
        cnn_size_t in_height,
        cnn_size_t window_width,
        cnn_size_t window_height,
        cnn_size_t in_channels,
        cnn_size_t out_channels,
        padding pad_type = padding::valid,
        bool has_bias = true,
        cnn_size_t w_stride = 1,
        cnn_size_t h_stride = 1)
        : Base(in_width * in_height * in_channels, conv_out_dim(in_width, in_height, window_width, window_height, w_stride, h_stride, pad_type) * out_channels,
            window_width*window_height * in_channels * out_channels, has_bias ? out_channels : 0),
        in_(in_width, in_height, in_channels),
        in_padded_(in_length(in_width, window_width, pad_type), in_length(in_height, window_height, pad_type), in_channels),
        out_(conv_out_length(in_width, window_width, w_stride, pad_type), conv_out_length(in_height, window_height, h_stride, pad_type), out_channels),
        weight_(window_width, window_height, in_channels*out_channels),
        pad_type_(pad_type),
        w_stride_(w_stride), h_stride_(h_stride)
    {
        init();
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
    **/
    convolutional_layer(cnn_size_t in_width,
        cnn_size_t in_height,
        cnn_size_t window_size,
        cnn_size_t in_channels,
        cnn_size_t out_channels,
        const connection_table& connection_table,
        padding pad_type = padding::valid,
        bool has_bias = true,
        cnn_size_t w_stride = 1,
        cnn_size_t h_stride = 1
        )
        : Base(in_width * in_height * in_channels, conv_out_dim(in_width, in_height, window_size, w_stride, h_stride, pad_type) * out_channels,
            sqr(window_size) * in_channels * out_channels, has_bias ? out_channels : 0),
        tbl_(connection_table),
        in_(in_width, in_height, in_channels),
        in_padded_(in_length(in_width, window_size, pad_type), in_length(in_height, window_size, pad_type), in_channels),
        out_(conv_out_length(in_width, window_size, w_stride, pad_type), conv_out_length(in_height, window_size, h_stride, pad_type), out_channels),
        weight_(window_size, window_size, in_channels*out_channels),
        pad_type_(pad_type),
        w_stride_(w_stride), h_stride_(h_stride)
    {
        init();
    }

    /**
    * constructing convolutional layer
    *
    * @param in_width         [in] input image width
    * @param in_height        [in] input image height
    * @param window_width  [in] window_width(kernel) size of convolution
    * @param window_height [in] window_height(kernel) size of convolution
    * @param in_channels      [in] input image channels (grayscale=1, rgb=3)
    * @param out_channels     [in] output image channels
    * @param connection_table [in] definition of connections between in-channels and out-channels
    * @param pad_type         [in] rounding strategy
    *                               valid: use valid pixels of input only. output-size = (in-width - window_size + 1) * (in-height - window_size + 1) * out_channels
    *                               same: add zero-padding to keep same width/height. output-size = in-width * in-height * out_channels
    **/
    convolutional_layer(cnn_size_t in_width,
        cnn_size_t in_height,
        cnn_size_t window_width,
        cnn_size_t window_height,
        cnn_size_t in_channels,
        cnn_size_t out_channels,
        const connection_table& connection_table,
        padding pad_type = padding::valid,
        bool has_bias = true,
        cnn_size_t w_stride = 1,
        cnn_size_t h_stride = 1
        )
        : Base(in_width * in_height * in_channels, conv_out_dim(in_width, in_height, window_width, window_height, w_stride, h_stride, pad_type) * out_channels,
            window_width*window_height * in_channels * out_channels, has_bias ? out_channels : 0),
        tbl_(connection_table),
        in_(in_width, in_height, in_channels),
        in_padded_(in_length(in_width, window_width, pad_type), in_length(in_height, window_height, pad_type), in_channels),
        out_(conv_out_length(in_width, window_width, w_stride, pad_type), conv_out_length(in_height, window_height, h_stride, pad_type), out_channels),
        weight_(window_width, window_height, in_channels*out_channels),
        pad_type_(pad_type),
        w_stride_(w_stride), h_stride_(h_stride)
    {
        init();
    }

    ///< number of incoming connections for each output unit
    virtual size_t fan_in_size() const override
    {
        return weight_.width_ * weight_.height_ * in_.depth_;
    }

    ///< number of outgoing connections for each input unit
    virtual size_t fan_out_size() const override
    {
        return (weight_.width_ / w_stride_) * (weight_.height_ / h_stride_) * out_.depth_;
    }

    ///< number of connections
    virtual size_t connection_size() const override
    {
        return out_.size() * fan_in_size();
    }

    virtual const vec_t& back_propagation_2nd(const vec_t& current_delta2) override
    {
        const vec_t& prev_out = *(prev_out_padded_[0]);
        const activation::function& prev_h = prev_->activation_function();
        vec_t* prev_delta = (pad_type_ == padding::same) ? &prev_delta2_padded_ : &prev_delta2_;

        std::fill(prev_delta->begin(), prev_delta->end(), float_t(0));

        // accumulate dw
        for_i(in_.depth_, [&](int inc) {
            for (cnn_size_t outc = 0; outc < out_.depth_; outc++) {

                if (!tbl_.is_connected(outc, inc)) continue;

                for (cnn_size_t wy = 0; wy < weight_.height_; wy++) {
                    for (cnn_size_t wx = 0; wx < weight_.width_; wx++) {
                        float_t dst = float_t(0);
                        const float_t * prevo = &prev_out[in_padded_.get_index(wx, wy, inc)];
                        const float_t * delta = &current_delta2[out_.get_index(0, 0, outc)];

                        for (cnn_size_t y = 0; y < out_.height_; y++) {
                            for (cnn_size_t x = 0; x < out_.width_; x++) {
                                dst += sqr(prevo[y * in_padded_.width_ + x]) * delta[y * out_.width_ + x];
                            }
                        }
                        Whessian_[weight_.get_index(wx, wy, in_.depth_ * outc + inc)] += dst;
                    }
                }
            }
        });

        // accumulate db
        if (!this->bhessian_.empty()) {
            for (cnn_size_t outc = 0; outc < out_.depth_; outc++) {
                const float_t *delta = &current_delta2[out_.get_index(0, 0, outc)];
                this->bhessian_[outc] += std::accumulate(delta, delta + out_.width_ * out_.height_, float_t(0));
            }
        }

        // propagate delta to previous layer
        for_i(in_.depth_, [&](int inc) {
            for (cnn_size_t outc = 0; outc < out_.depth_; outc++) {
                if (!tbl_.is_connected(outc, inc)) continue;

                const float_t *pw = &W_[weight_.get_index(0, 0, in_.depth_ * outc + inc)];
                const float_t *pdelta_src = &current_delta2[out_.get_index(0, 0, outc)];
                float_t *pdelta_dst = &(*prev_delta)[in_padded_.get_index(0, 0, inc)];

                for (cnn_size_t y = 0; y < out_.height_; y++) {
                    for (cnn_size_t x = 0; x < out_.width_; x++) {
                        const float_t * ppw = pw;
                        const float_t ppdelta_src = pdelta_src[y * out_.width_ + x];
                        float_t * ppdelta_dst = pdelta_dst + y * h_stride_ * in_padded_.width_ + x * w_stride_;

                        for (cnn_size_t wy = 0; wy < weight_.height_; wy++) {
                            for (cnn_size_t wx = 0; wx < weight_.width_; wx++) {
                                ppdelta_dst[wy * in_padded_.width_ + wx] += sqr(*ppw++) * ppdelta_src;
                            }
                        }
                    }
                }
            }
        });

        for_i(parallelize_, in_padded_.size(), [&](int i) {
            (*prev_delta)[i] *= sqr(prev_h.df(prev_out[i]));
        });

        if (pad_type_ == padding::same)
            copy_and_unpad_delta(prev_delta2_padded_, prev_delta2_);

        CNN_LOG_VECTOR(current_delta2, "[pc]curr-delta2");
        CNN_LOG_VECTOR(prev_delta2_, "[pc]prev-delta2");
        CNN_LOG_VECTOR(Whessian_, "[pc]whessian");

        return prev_->back_propagation_2nd(prev_delta2_);
    }

    virtual const vec_t& forward_propagation(const vec_t& in_raw, size_t worker_index) override
    {
        copy_and_pad_input(in_raw, static_cast<int>(worker_index));

        vec_t &a = a_[worker_index]; // w*x
        vec_t &out = output_[worker_index]; // output
        const vec_t &in = *(prev_out_padded_[worker_index]); // input
        
        std::fill(a.begin(), a.end(), float_t(0));

        for_i(parallelize_, out_.depth_, [&](int o) {
            for (cnn_size_t inc = 0; inc < in_.depth_; inc++) {
                if (!tbl_.is_connected(o, inc)) continue;

                const float_t *pw = &this->W_[weight_.get_index(0, 0, in_.depth_ * o + inc)];
                const float_t *pi = &in[in_padded_.get_index(0, 0, inc)];
                float_t *pa = &a[out_.get_index(0, 0, o)];

                for (cnn_size_t y = 0; y < out_.height_; y++) {
                    for (cnn_size_t x = 0; x < out_.width_; x++) {
                        const float_t * ppw = pw;
                        const float_t * ppi = pi + (y * h_stride_) * in_padded_.width_ + x * w_stride_;
                        float_t sum = float_t(0);

                        // should be optimized for small kernel(3x3,5x5)
                        for (cnn_size_t wy = 0; wy < weight_.height_; wy++) {
                            for (cnn_size_t wx = 0; wx < weight_.width_; wx++) {
                                sum += *ppw++ * ppi[wy * in_padded_.width_ + wx];
                            }
                        }
                        pa[y * out_.width_ + x] += sum;
                    }
                }
            }

            if (!this->b_.empty()) {
                float_t *pa = &a[out_.get_index(0, 0, o)];
                float_t b = this->b_[o];
                std::for_each(pa, pa + out_.width_ * out_.height_, [&](float_t& f) { f += b; });
            }
        });

        for_i(parallelize_, out_size_, [&](int i) {
            out[i] = h_.f(a, i);
        });

        CNN_LOG_VECTOR(in_raw, "[pc]in");
        CNN_LOG_VECTOR(W_, "[pc]w");
        CNN_LOG_VECTOR(a, "[pc]a");
        CNN_LOG_VECTOR(out, "[pc]forward");

        return next_ ? next_->forward_propagation(out, worker_index) : out;
    }

    float_t& weight_at(cnn_size_t in_channel, cnn_size_t out_channel, cnn_size_t kernel_x, cnn_size_t kernel_y) {
        return W_[weight_.get_index(kernel_x, kernel_y, in_.depth_ * out_channel + in_channel)];
    }

    const vec_t& back_propagation(const vec_t& curr_delta, size_t index) override {
        const vec_t& prev_out = *(prev_out_padded_[index]);
        const activation::function& prev_h = prev_->activation_function();
        vec_t* prev_delta = (pad_type_ == padding::same) ? &prev_delta_padded_[index] : &prev_delta_[index];
        vec_t& dW = dW_[index];
        vec_t& db = db_[index];

        std::fill(prev_delta->begin(), prev_delta->end(), float_t(0));

        // propagate delta to previous layer
        for_i(in_.depth_, [&](int inc) {
            for (cnn_size_t outc = 0; outc < out_.depth_; outc++) {
                if (!tbl_.is_connected(outc, inc)) continue;

                const float_t *pw = &this->W_[weight_.get_index(0, 0, in_.depth_ * outc + inc)];
                const float_t *pdelta_src = &curr_delta[out_.get_index(0, 0, outc)];
                float_t *pdelta_dst = &(*prev_delta)[in_padded_.get_index(0, 0, inc)];

                for (cnn_size_t y = 0; y < out_.height_; y++) {
                    for (cnn_size_t x = 0; x < out_.width_; x++) {
                        const float_t * ppw = pw;
                        const float_t ppdelta_src = pdelta_src[y * out_.width_ + x];
                        float_t * ppdelta_dst = pdelta_dst + y * h_stride_ * in_padded_.width_ + x * w_stride_;

                        for (cnn_size_t wy = 0; wy < weight_.height_; wy++) {
                            for (cnn_size_t wx = 0; wx < weight_.width_; wx++) {
                                ppdelta_dst[wy * in_padded_.width_ + wx] += *ppw++ * ppdelta_src;
                            }
                        }
                    }
                }
            }
        });

        for_i(parallelize_, in_padded_.size(), [&](int i) {
            (*prev_delta)[i] *= prev_h.df(prev_out[i]);
        });

        // accumulate dw
        for_i(in_.depth_, [&](int inc) {
            for (cnn_size_t outc = 0; outc < out_.depth_; outc++) {

                if (!tbl_.is_connected(outc, inc)) continue;

                for (cnn_size_t wy = 0; wy < weight_.height_; wy++) {
                    for (cnn_size_t wx = 0; wx < weight_.width_; wx++) {
                        float_t dst = float_t(0);
                        const float_t * prevo = &prev_out[in_padded_.get_index(wx, wy, inc)];
                        const float_t * delta = &curr_delta[out_.get_index(0, 0, outc)];

                        for (cnn_size_t y = 0; y < out_.height_; y++) {
                            dst += vectorize::dot(prevo + y * in_padded_.width_, delta + y * out_.width_, out_.width_);
                        }
                        dW[weight_.get_index(wx, wy, in_.depth_ * outc + inc)] += dst;
                    }
                }
            }
        });

        // accumulate db
        if (!db.empty()) {
            for (cnn_size_t outc = 0; outc < out_.depth_; outc++) {
                const float_t *delta = &curr_delta[out_.get_index(0, 0, outc)];
                db[outc] += std::accumulate(delta, delta + out_.width_ * out_.height_, float_t(0));
            }
        }

        if (pad_type_ == padding::same)
            copy_and_unpad_delta(prev_delta_padded_[index], prev_delta_[index]);

        CNN_LOG_VECTOR(curr_delta, "[pc]curr_delta");
        CNN_LOG_VECTOR(prev_delta_[index], "[pc]prev_delta");
        CNN_LOG_VECTOR(dW, "[pc]dW");
        CNN_LOG_VECTOR(db, "[pc]db");

        return prev_->back_propagation(prev_delta_[index], index);
    }

    index3d<cnn_size_t> in_shape() const override { return in_; }
    index3d<cnn_size_t> out_shape() const override { return out_; }
    std::string layer_type() const override { return "conv"; }

    image<> weight_to_image() const {
        image<> img;
        const cnn_size_t border_width = 1;
        const auto pitch = weight_.width_ + border_width;
        const auto width = out_.depth_ * pitch + border_width;
        const auto height = in_.depth_ * pitch + border_width;
        const image<>::intensity_t bg_color = 255;

        img.resize(width, height);
        img.fill(bg_color);

        auto minmax = std::minmax_element(this->W_.begin(), this->W_.end());

        for (cnn_size_t r = 0; r < in_.depth_; ++r) {
            for (cnn_size_t c = 0; c < out_.depth_; ++c) {
                if (!tbl_.is_connected(c, r)) continue;

                const auto top = r * pitch + border_width;
                const auto left = c * pitch + border_width;

                for (cnn_size_t y = 0; y < weight_.height_; ++y) {
                    for (cnn_size_t x = 0; x < weight_.width_; ++x) {
                        const float_t w = W_[weight_.get_index(x, y, c * in_.depth_ + r)];

                        img.at(left + x, top + y)
                            = static_cast<image<>::intensity_t>(rescale(w, *minmax.first, *minmax.second, 0, 255));
                    }
                }
            }
        }
        return img;
    }

private:
    void init() {
        for (cnn_size_t i = 0; i < CNN_TASK_SIZE; i++) {
            if (pad_type_ == padding::same) {
                prev_out_buf_[i] = new vec_t(in_padded_.size(), float_t(0));
                prev_delta_padded_[i].resize(in_padded_.size(), float_t(0));               
            }
            else {
                prev_out_buf_[i] = nullptr;
            }
        }
        if (pad_type_ == padding::same) {
            prev_delta2_padded_.resize(in_padded_.size(), float_t(0));
        }
    }

    cnn_size_t in_length(cnn_size_t in_length, cnn_size_t window_size, padding pad_type) const {
        return pad_type == padding::same ? (in_length + window_size - 1) : in_length;
    }

    static cnn_size_t conv_out_length(cnn_size_t in_length, cnn_size_t window_size, cnn_size_t stride, padding pad_type) {
        return pad_type == padding::same ? (cnn_size_t)ceil((double)in_length / stride) : (cnn_size_t)ceil((double)(in_length - window_size + 1) / stride);
    }

    static cnn_size_t conv_out_dim(cnn_size_t in_width, cnn_size_t in_height, cnn_size_t window_size, cnn_size_t w_stride, cnn_size_t h_stride, padding pad_type) {
        return conv_out_length(in_width, window_size, w_stride, pad_type) * conv_out_length(in_height, window_size, h_stride, pad_type);
    }

    cnn_size_t conv_out_dim(cnn_size_t in_width, cnn_size_t in_height, cnn_size_t window_width, cnn_size_t window_height, cnn_size_t w_stride, cnn_size_t h_stride, padding pad_type) const {
        return conv_out_length(in_width, window_width, w_stride, pad_type) * conv_out_length(in_height, window_height, h_stride, pad_type);
    }

    void copy_and_unpad_delta(const vec_t& delta, vec_t& dst) {
        if (pad_type_ == padding::valid) {
            dst = delta;
        }
        else {
            for (cnn_size_t c = 0; c < in_.depth_; c++) {
                float_t *pdst = &dst[in_.get_index(0, 0, c)];
                const float_t *pin = &delta[in_padded_.get_index(weight_.width_ / 2, weight_.height_ / 2, c)];

                for (cnn_size_t y = 0; y < in_.height_; y++, pdst += in_.width_, pin += in_padded_.width_) {
                    std::copy(pin, pin + in_.width_, pdst);
                }
            }
        }
    }

    void copy_and_pad_input(const vec_t& in, int worker_index) {
        vec_t* dst = prev_out_buf_[worker_index];

        if (pad_type_ == padding::valid) {
            prev_out_padded_[worker_index] = &in;
        }
        else {
            // make padded version in order to avoid corner-case in fprop/bprop
            for (cnn_size_t c = 0; c < in_.depth_; c++) {
                float_t *pimg = &(*dst)[in_padded_.get_index(weight_.width_ / 2, weight_.height_ / 2, c)];
                const float_t *pin = &in[in_.get_index(0, 0, c)];

                for (cnn_size_t y = 0; y < in_.height_; y++, pin += in_.width_, pimg += in_padded_.width_) {
                    std::copy(pin, pin + in_.width_, pimg);
                }
            }
            prev_out_padded_[worker_index] = prev_out_buf_[worker_index];
        }
    }

    const vec_t* prev_out_padded_[CNN_TASK_SIZE];
    vec_t* prev_out_buf_[CNN_TASK_SIZE];
    vec_t  prev_delta_padded_[CNN_TASK_SIZE];
    vec_t  prev_delta2_padded_;

    connection_table tbl_;
    index3d<cnn_size_t> in_;
    index3d<cnn_size_t> in_padded_;
    index3d<cnn_size_t> out_;
    index3d<cnn_size_t> weight_;
    padding pad_type_;
    size_t w_stride_;
    size_t h_stride_;
};

#if 0

#include "tiny_cnn/layers/partial_connected_layer.h"

template<typename Activation = activation::identity>
class convolutional_layer : public partial_connected_layer<Activation> {
public:
    typedef partial_connected_layer<Activation> Base;
    CNN_USE_LAYER_MEMBERS;

    /**
     * constructing convolutional layer
     *
     * @param in_width     [in] input image width
     * @param in_height    [in] input image height
     * @param window_size  [in] window(kernel) size of convolution
     * @param in_channels  [in] input image channels (grayscale=1, rgb=3)
     * @param out_channels [in] output image channels
     * @param padding      [in] rounding strategy
     *                          valid: use valid pixels of input only. output-size = (in-width - window_size + 1) * (in-height - window_size + 1) * out_channels
     *                          same: add zero-padding to keep same width/height. output-size = in-width * in-height * out_channels
     **/
    convolutional_layer_old(cnn_size_t in_width,
                        cnn_size_t in_height,
                        cnn_size_t window_size,
                        cnn_size_t in_channels,
                        cnn_size_t out_channels,
                        padding pad_type = padding::valid)
    : Base(in_width * in_height * in_channels, out_size(in_width, in_height, window_size, pad_type) * out_channels, 
           sqr(window_size) * in_channels * out_channels, out_channels), 
      in_(in_width, in_height, in_channels), 
      out_(out_length(in_width, window_size, pad_type), out_length(in_height, window_size, pad_type), out_channels),
      weight_(window_size, window_size, in_channels*out_channels),
      window_size_(window_size)
    {
        init_connection(connection_table(), pad_type);
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
     **/
    convolutional_layer_old(cnn_size_t in_width,
                        cnn_size_t in_height,
                        cnn_size_t window_size,
                        cnn_size_t in_channels,
                        cnn_size_t out_channels,
                        const connection_table& connection_table,
                        padding pad_type = padding::valid)
        : Base(in_width * in_height * in_channels, out_size(in_width, in_height, window_size, pad_type) * out_channels, 
               sqr(window_size) * in_channels * out_channels, out_channels), 
          in_(in_width, in_height, in_channels), 
          out_(out_length(in_width, window_size, pad_type), out_length(in_height, window_size, pad_type), out_channels),
          weight_(window_size, window_size, in_channels*out_channels),
          connection_(connection_table),
          window_size_(window_size)
    {
        init_connection(connection_table, pad_type);
        //this->remap();
    }

    image<> output_to_image(size_t worker_index = 0) const {
        return vec2image<unsigned char>(output_[worker_index], out_);
    }

    image<> weight_to_image() const {
        image<> img;
        const cnn_size_t border_width = 1;
        const auto pitch = window_size_ + border_width;
        const auto width = out_.depth_ * pitch + border_width;
        const auto height = in_.depth_ * pitch + border_width;
        const image<>::intensity_t bg_color = 255;

        img.resize(width, height);
        img.fill(bg_color);

        auto minmax = std::minmax_element(this->W_.begin(), this->W_.end());

        for (cnn_size_t r = 0; r < in_.depth_; ++r) {
            for (cnn_size_t c = 0; c < out_.depth_; ++c) {
                if (!connection_.is_connected(c, r)) continue;

                const auto top = r * pitch + border_width;
                const auto left = c * pitch + border_width;

                for (cnn_size_t y = 0; y < window_size_; ++y) {
                    for (cnn_size_t x = 0; x < window_size_; ++x) {
                        const float_t w = W_[weight_.get_index(x, y, c * in_.depth_ + r)];

                        img.at(left + x, top + y)
                            = static_cast<image<>::intensity_t>(rescale(w, *minmax.first, *minmax.second, 0, 255));
                    }
                }
            }
        }
        return img;
    }

    index3d<cnn_size_t> in_shape() const override { return in_; }
    index3d<cnn_size_t> out_shape() const override { return out_; }
    std::string layer_type() const override { return "conv"; }

private:
    cnn_size_t out_length(cnn_size_t in_length, cnn_size_t window_size, padding pad_type) const {
        return pad_type == padding::same ? in_length : (in_length - window_size + 1);
    }

    cnn_size_t out_size(cnn_size_t in_width, cnn_size_t in_height, cnn_size_t window_size, padding pad_type) const {
        return out_length(in_width, window_size, pad_type) * out_length(in_height, window_size, pad_type);
    }

    void init_connection(const connection_table& table, padding pad_type) {
        cnn_size_t pad = (pad_type == padding::valid) ? 0 : window_size_ / 2;

        for (cnn_size_t inc = 0; inc < in_.depth_; ++inc) {
            for (cnn_size_t outc = 0; outc < out_.depth_; ++outc) {
                if (!table.is_connected(outc, inc)) {
                    continue;
                }

                for (cnn_size_t y = 0; y < out_.height_; ++y)
                    for (cnn_size_t x = 0; x < out_.width_; ++x)
                        connect_kernel(inc, outc, x, y, pad);
            }
        }

        for (cnn_size_t outc = 0; outc < out_.depth_; ++outc)
            for (cnn_size_t y = 0; y < out_.height_; ++y)
                for (cnn_size_t x = 0; x < out_.width_; ++x)
                    this->connect_bias(outc, out_.get_index(x, y, outc));
    }

    void connect_kernel(cnn_size_t inc, cnn_size_t outc, cnn_size_t x, cnn_size_t y, cnn_size_t pad) {

        for (cnn_size_t dy = 0; dy < window_size_; ++dy) {
            if (y + dy < pad) continue;
            if (y + dy - pad >= in_.height_) continue;

            for (cnn_size_t dx = 0; dx < window_size_; ++dx) {
                if (x + dx < pad) continue;
                if (x + dx - pad >= in_.width_) continue;

                this->connect_weight(
                    in_.get_index(x + dx - pad, y + dy - pad, inc), 
                    out_.get_index(x, y, outc), 
                    weight_.get_index(dx, dy, outc * in_.depth_ + inc));
            }
        }
    }

    index3d<cnn_size_t> in_;
    index3d<cnn_size_t> out_;
    index3d<cnn_size_t> weight_;
    connection_table connection_;
    size_t window_size_;
};

#endif

} // namespace tiny_cnn
