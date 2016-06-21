/*
    Copyright (c) 2013, Taiga Nomi
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

#include "tiny_cnn/core/tiny_backend.h"
#include "tiny_cnn/core/nnp_backend.h"
#include "tiny_cnn/core/dnn_backend.h"

#include "tiny_cnn/layers/base_conv_layer.h"

#ifdef CNN_USE_NNPACK
#include "nnpack.h"
#endif

/* DEPRECATED CLASS */
/* DEPRECATED CLASS */
/* DEPRECATED CLASS */
/* DEPRECATED CLASS */
/* DEPRECATED CLASS */
/* DEPRECATED CLASS */
/* DEPRECATED CLASS */
/* DEPRECATED CLASS */

namespace tiny_cnn {

using namespace core;

/**
 * 2D convolution layer
 *
 * take input as two-dimensional *image* and applying filtering operation.
 **/
template<typename Activation = activation::identity>
class convolutional_layer : public base_conv_layer<Activation> {
 public:
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
    * @param has_bias     [in] whether to add a bias vector to the filter outputs
    * @param w_stride     [in] specify the horizontal interval at which to apply the filters to the input
    * @param h_stride     [in] specify the vertical interval at which to apply the filters to the input
    **/
    convolutional_layer(cnn_size_t     in_width,
                        cnn_size_t     in_height,
                        cnn_size_t     window_size,
                        cnn_size_t     in_channels,
                        cnn_size_t     out_channels,
                        padding        pad_type = padding::valid,
                        bool           has_bias = true,
                        cnn_size_t     w_stride = 1,
                        cnn_size_t     h_stride = 1,
                        backend_t      backend_type = backend_t::tiny_cnn,
                        backend_params backend_params = backend_params())
        : base_conv_layer<Activation>(in_width, in_height, window_size,
                                      in_channels, out_channels,
                                      pad_type, has_bias, w_stride, h_stride) {}

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
    convolutional_layer(cnn_size_t     in_width,
                        cnn_size_t     in_height,
                        cnn_size_t     window_width,
                        cnn_size_t     window_height,
                        cnn_size_t     in_channels,
                        cnn_size_t     out_channels,
                        padding        pad_type = padding::valid,
                        bool           has_bias = true,
                        cnn_size_t     w_stride = 1,
                        cnn_size_t     h_stride = 1,
                        backend_t      backend_type = backend_t::tiny_cnn,
                        backend_params backend_params = backend_params())
        : base_conv_layer<Activation>(in_width, in_height,
                                      window_width, window_height,
                                      in_channels, out_channels,
                                      pad_type, has_bias, w_stride, h_stride) {
        switch (backend_type) {
            case backend_t::tiny_cnn: this->backend_ = std::make_shared<core::tiny_backend>();
            case backend_t::nnpack:  this->backend_ = std::make_shared<core::nnp_backend>();
            case backend_t::libdnn:  this->backend_ = std::make_shared<core::dnn_backend>();
            default: nn_error("not supported backed type");
        }

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
    convolutional_layer(cnn_size_t              in_width,
                        cnn_size_t              in_height,
                        cnn_size_t              window_size,
                        cnn_size_t              in_channels,
                        cnn_size_t              out_channels,
                        const connection_table& connection_table,
                        padding                 pad_type = padding::valid,
                        bool                    has_bias = true,
                        cnn_size_t              w_stride = 1,
                        cnn_size_t              h_stride = 1,
                        backend_t      backend_type = backend_t::tiny_cnn,
                        backend_params backend_params = backend_params())
        : base_conv_layer<Activation>(in_width, in_height, window_size,
                                      in_channels, out_channels,
                                      connection_table,
                                      pad_type, has_bias, w_stride, h_stride) {}

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
    convolutional_layer(cnn_size_t              in_width,
                        cnn_size_t              in_height,
                        cnn_size_t              window_width,
                        cnn_size_t              window_height,
                        cnn_size_t              in_channels,
                        cnn_size_t              out_channels,
                        const connection_table& connection_table,
                        padding                 pad_type = padding::valid,
                        bool                    has_bias = true,
                        cnn_size_t              w_stride = 1,
                        cnn_size_t              h_stride = 1,
                        backend_t   backend_type = backend_t::tiny_cnn,
                        backend_params backend_params = backend_params())
        : base_conv_layer<Activation>(in_width, in_height,
                                      window_width, window_height,
                                      in_channels, out_channels,
                                      connection_table,
                                      pad_type, has_bias, w_stride, h_stride) {}

    void forward_propagation(cnn_size_t index,
                             const std::vector<vec_t*>& in_data,
                             std::vector<vec_t*>& out_data) {
        /*if ((nnpack_supported() && this->has_bias_ ) ||
            (nnpack_supported() && (this->w_stride_ == 1 &&
                                    this->h_stride_ == 1))) {
            nnp_forward_propagation(index, in_data, out_data);
        } else {
            tiny_forward_propagation(index, in_data, out_data);
        }*/
        tiny_forward_propagation(index, in_data, out_data);
    }

    void back_propagation(cnn_size_t                 index,
                          const std::vector<vec_t*>& in_data,
                          const std::vector<vec_t*>& out_data,
                          std::vector<vec_t*>&       out_grad,
                          std::vector<vec_t*>&       in_grad) {
        tiny_back_propagation(index, in_data, out_data, out_grad, in_grad);
    }

 private:
    void tiny_forward_propagation(cnn_size_t index,
                                  const std::vector<vec_t*>& in_data,
                                  std::vector<vec_t*>& out_data) {
        this->copy_and_pad_input(*in_data[0], static_cast<int>(index));
        const vec_t& W   = *in_data[1];
        vec_t&       out = *out_data[0];
        vec_t&       a   = *out_data[1];
        const vec_t &in  = *(this->conv_layer_worker_storage_[index].prev_out_padded_); // input //NOLINT

        cnn_size_t idx = 0;
        std::fill(a.begin(), a.end(), float_t(0));

        for_i(this->parallelize_, this->out_.depth_, [&](int o) {
            for (cnn_size_t inc = 0; inc < this->in_.depth_; inc++) {
                if (!this->tbl_.is_connected(o, inc)) continue;

                idx = this->in_.depth_ * o + inc;
                idx = this->weight_.get_index(0, 0, idx);
                const float_t *pw = &W[idx];

                idx = this->in_padded_.get_index(0, 0, inc);
                const float_t *pi = &in[idx];

                idx = this->out_.get_index(0, 0, o);
                float_t *pa = &a[idx];

                for (cnn_size_t y = 0; y < this->out_.height_; y++) {
                    for (cnn_size_t x = 0; x < this->out_.width_; x++) {
                        const float_t * ppw = pw;
                        const float_t * ppi = pi + this->in_padded_.width_ *
                                              (y * this->h_stride_) +
                                               x * this->w_stride_;
                        float_t sum = float_t(0);

                        // should be optimized for small kernel(3x3,5x5)
                        for (cnn_size_t wy = 0; wy < this->weight_.height_; wy++) {    //NOLINT
                            for (cnn_size_t wx = 0; wx < this->weight_.width_; wx++) { //NOLINT
                                idx = wy * this->in_padded_.width_ + wx;
                                sum += *ppw++ * ppi[idx];
                            }
                        }
                        pa[y * this->out_.width_ + x] += sum;
                    }
                }
            }

            if (this->has_bias_) {
                const vec_t& bias = *in_data[2];
                float_t * pa  = &a[this->out_.get_index(0, 0, o)];
                float_t * paa = pa + this->out_.width_ * this->out_.height_;
                std::for_each(pa, paa, [&](float_t& f) { f += bias[o]; });
            }
        });

        for_i(this->parallelize_, this->out_.size(), [&](int i) {
            out[i] = this->h_.f(a, i);
        });
    }

    float_t& weight_at(cnn_size_t in_channel, cnn_size_t out_channel,
                       cnn_size_t kernel_x, cnn_size_t kernel_y) {
        vec_t* W = this->get_weights()[0];
        cnn_size_t kernel_tmp  = this->in_.depth_ * out_channel + in_channel;
        return W[this->weight_.get_index(kernel_x, kernel_y, kernel_tmp)];
    }

    void tiny_back_propagation(cnn_size_t                 index,
                               const std::vector<vec_t*>& in_data,
                               const std::vector<vec_t*>& out_data,
                               std::vector<vec_t*>&       out_grad,
                               std::vector<vec_t*>&       in_grad) {
        conv_layer_worker_specific_storage& cws =
            this->conv_layer_worker_storage_[index];

        const vec_t& prev_out = *(cws.prev_out_padded_);
        const vec_t& W = *in_data[1];
        vec_t&       dW = *in_grad[1];
        vec_t&       curr_delta = *out_grad[1];
        vec_t*       prev_delta = (this->pad_type_ == padding::same) ?
                                  &cws.prev_delta_padded_ : in_grad[0];

        assert(W.size() == this->weight_.size());
        assert(dW.size() == this->weight_.size());
        assert(curr_delta.size() ==  this->out_shape()[0].size());

        this->backward_activation(*out_grad[0], *out_data[0], curr_delta);

        cnn_size_t idx = 0;
        std::fill(prev_delta->begin(), prev_delta->end(), float_t(0));

        // propagate delta to previous layer
        for_i(this->in_.depth_, [&](int inc) {
            for (cnn_size_t outc = 0; outc < this->out_.depth_; outc++) {
                if (!this->tbl_.is_connected(outc, inc)) continue;

                idx = this->in_.depth_ * outc + inc;
                idx = this->weight_.get_index(0, 0, idx);
                const float_t *pw = &W[idx];

                idx = this->out_.get_index(0, 0, outc);
                const float_t *pdelta_src = &curr_delta[idx];

                idx = this->in_padded_.get_index(0, 0, inc);
                float_t *pdelta_dst = &(*prev_delta)[idx];

                for (cnn_size_t y = 0; y < this->out_.height_; y++) {
                    for (cnn_size_t x = 0; x < this->out_.width_; x++) {
                        const float_t * ppw = pw;

                        idx = y * this->out_.width_ + x;
                        const float_t ppdelta_src = pdelta_src[idx];

                        float_t * ppdelta_dst = pdelta_dst +
                            y * this->h_stride_ * this->in_padded_.width_ +
                            x * this->w_stride_;

                        for (cnn_size_t wy = 0; wy < this->weight_.height_; wy++) {    // NOLINT
                            for (cnn_size_t wx = 0; wx < this->weight_.width_; wx++) { // NOLINT
                                idx = wy * this->in_padded_.width_ + wx;
                                ppdelta_dst[idx] += *ppw++ * ppdelta_src;
                            }
                        }
                    }
                }
            }
        });

        // accumulate dw
        for_i(this->in_.depth_, [&](int inc) {
            for (cnn_size_t outc = 0; outc < this->out_.depth_; outc++) {
                if (!this->tbl_.is_connected(outc, inc)) continue;

                for (cnn_size_t wy = 0; wy < this->weight_.height_; wy++) {
                    for (cnn_size_t wx = 0; wx < this->weight_.width_; wx++) {
                        float_t dst = float_t(0);

                        idx = this->in_padded_.get_index(wx, wy, inc);
                        const float_t * prevo = &prev_out[idx];

                        idx = this->out_.get_index(0, 0, outc);
                        const float_t * delta = &curr_delta[idx];

                        for (cnn_size_t y = 0; y < this->out_.height_; y++) {
                            dst += vectorize::dot(
                                prevo + y * this->in_padded_.width_,
                                delta + y * this->out_.width_,
                                this->out_.width_);
                        }

                        idx = this->in_.depth_ * outc + inc;
                        dW[this->weight_.get_index(wx, wy, idx)] += dst;
                    }
                }
            }
        });

        // accumulate db
        if (this->has_bias_) {
            vec_t& db = *in_grad[2];

            for (cnn_size_t outc = 0; outc < this->out_.depth_; outc++) {
                idx = this->out_.get_index(0, 0, outc);
                const float_t * delta = &curr_delta[idx];
                const float_t * deltaa = delta + this->out_.width_ *
                                                 this->out_.height_;
                db[outc] += std::accumulate(delta, deltaa, float_t(0));
            }
        }

        if (this->pad_type_ == padding::same) {
            this->copy_and_unpad_delta(cws.prev_delta_padded_, *in_grad[0]);
        }
    }

    void nnp_forward_propagation(cnn_size_t index,
                                 const std::vector<vec_t*>& in_data,
                                 std::vector<vec_t*>& out_data) {
#ifdef CNN_USE_NNPACK
        // TODO: auto configure algo
        //const auto algorithm = nnp_algorithm(algo);
        const nnp_convolution_algorithm algorithm = nnp_convolution_algorithm_auto;

        // TODO: auto configure kernel
        // const auto kernel_transform_strategy = nnp_kts(kts);
        const nnp_convolution_kernel_transform_strategy kernel_tf_strategy =
            nnp_convolution_kernel_transform_strategy_reuse;

        const cnn_size_t input_channels = this->in_.depth_;
        const cnn_size_t output_channels = this->out_.depth_;

        const nnp_size input_size = {
            static_cast<size_t>(this->in_.width_),
            static_cast<size_t>(this->in_.height_)
        };

        const nnp_size kernel_size = {
            static_cast<size_t>(this->weight_.width_),
            static_cast<size_t>(this->weight_.height_)
        };

        /*const nnp_padding padding = {.top = static_cast<size_t>(pad.cpu_data()[0]),
                                     .right = static_cast<size_t>(pad.cpu_data()[1]),
                                     .bottom = static_cast<size_t>(pad.cpu_data()[0]),
                                     .left = static_cast<size_t>(pad.cpu_data()[1])};*/
        const nnp_padding padding = { 0, 0, 0, 0 };

        const float* input_pointer =
            reinterpret_cast<const float*>(
                this->conv_layer_worker_storage_[index].prev_out_padded_);

        const float* kernel_pointer =
            reinterpret_cast<const float*>(&in_data[0]->at(0));

        const float* bias =
            reinterpret_cast<const float*>(&in_data[1]->at(0));

        float* output_pointer =
            reinterpret_cast<float*>(&out_data[0]->at(0));

        // TODO: embed it into a class
        const size_t num_mkl_threads = 1;
        pthreadpool_t threadpool = pthreadpool_create(num_mkl_threads);

        nnp_profile* profile = 0;

        //nnp_status status = nnp_convolution_inference(
        nnp_convolution_inference(
            algorithm,
            kernel_tf_strategy,
            input_channels,
            output_channels,
            input_size,
            padding,
            kernel_size,
            input_pointer,
            kernel_pointer,
            bias,
            output_pointer,
            threadpool,
            profile);

        // TODO: embed it into a class
        pthreadpool_destroy(threadpool);
#else
        nn_not_implemented_error();
#endif
    }

    void nnp_back_propagation(cnn_size_t                 index,
                              const std::vector<vec_t*>& in_data,
                              const std::vector<vec_t*>& out_data,
                              std::vector<vec_t*>&       out_grad,
                              std::vector<vec_t*>&       in_grad) {
        nn_not_implemented_error();
    }

};

}  // namespace tiny_cnn