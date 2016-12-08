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

#include <vector>
#include <string>
#include <algorithm>

#include "tiny_dnn/core/kernels/conv2d_op.h"
#include "tiny_dnn/core/kernels/conv2d_grad_op.h"
#include "tiny_dnn/core/kernels/conv2d_op_opencl.h"
#include "tiny_dnn/core/kernels/conv2d_op_libdnn.h"

#include "tiny_dnn/util/util.h"
#include "tiny_dnn/util/image.h"
#include "tiny_dnn/activations/activation_function.h"

using namespace tiny_dnn::core;

namespace tiny_dnn {

/**
 * 2D convolution layer
 *
 * take input as two-dimensional *image* and applying filtering operation.
 **/
template<typename Activation = activation::identity>
class convolutional_layer : public feedforward_layer<Activation> {
 public:
    typedef feedforward_layer<Activation> Base;
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
    *                          - valid: use valid pixels of input only. ```output-size = (in-width - window_width + 1) * (in-height - window_height + 1) * out_channels```
    *                          - same: add zero-padding to keep same width/height. ```output-size = in-width * in-height * out_channels```
    * @param has_bias     [in] whether to add a bias vector to the filter outputs
    * @param w_stride     [in] specify the horizontal interval at which to apply the filters to the input
    * @param h_stride     [in] specify the vertical interval at which to apply the filters to the input
    * @param backend_type [in] specify backend engine you use
    **/
    convolutional_layer(serial_size_t in_width,
                        serial_size_t in_height,
                        serial_size_t window_size,
                        serial_size_t in_channels,
                        serial_size_t out_channels,
                        padding    pad_type = padding::valid,
                        bool       has_bias = true,
                        serial_size_t w_stride = 1,
                        serial_size_t h_stride = 1,
                        backend_t  backend_type = core::default_engine())
        : convolutional_layer(in_width, in_height, window_size, window_size,
			      in_channels, out_channels, connection_table(),
			      pad_type, has_bias, w_stride, h_stride,
			      backend_type) {}

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
    *                           - valid: use valid pixels of input only. ```output-size = (in-width - window_width + 1) * (in-height - window_height + 1) * out_channels```
    *                           - same: add zero-padding to keep same width/height. ```output-size = in-width * in-height * out_channels```
    * @param has_bias     [in] whether to add a bias vector to the filter outputs
    * @param w_stride     [in] specify the horizontal interval at which to apply the filters to the input
    * @param h_stride     [in] specify the vertical interval at which to apply the filters to the input
    * @param backend_type [in] specify backend engine you use  
    **/
    convolutional_layer(serial_size_t in_width,
                        serial_size_t in_height,
                        serial_size_t window_width,
                        serial_size_t window_height,
                        serial_size_t in_channels,
                        serial_size_t out_channels,
                        padding    pad_type = padding::valid,
                        bool       has_bias = true,
                        serial_size_t w_stride = 1,
                        serial_size_t h_stride = 1,
                        backend_t  backend_type = core::default_engine())
        : convolutional_layer(in_width, in_height, window_width, window_height,
			      in_channels, out_channels, connection_table(),
			      pad_type, has_bias, w_stride, h_stride,
			      backend_type) {}

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
    *                              - valid: use valid pixels of input only. ```output-size = (in-width - window_width + 1) * (in-height - window_height + 1) * out_channels```
    *                              - same: add zero-padding to keep same width/height. ```output-size = in-width * in-height * out_channels```
    * @param has_bias         [in] whether to add a bias vector to the filter outputs
    * @param w_stride         [in] specify the horizontal interval at which to apply the filters to the input
    * @param h_stride         [in] specify the vertical interval at which to apply the filters to the input
    * @param backend_type [in] specify backend engine you use   
    **/
    convolutional_layer(serial_size_t              in_width,
                        serial_size_t              in_height,
                        serial_size_t              window_size,
                        serial_size_t              in_channels,
                        serial_size_t              out_channels,
                        const connection_table& connection_table,
                        padding                 pad_type = padding::valid,
                        bool                    has_bias = true,
                        serial_size_t              w_stride = 1,
                        serial_size_t              h_stride = 1,
                        backend_t      backend_type = core::default_engine())
        : convolutional_layer(in_width, in_height, window_size, window_size,
			      in_channels, out_channels, connection_table,
			      pad_type, has_bias, w_stride, h_stride,
                              backend_type) {}

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
    *                              - valid: use valid pixels of input only. ```output-size = (in-width - window_width + 1) * (in-height - window_height + 1) * out_channels```
    *                              - same: add zero-padding to keep same width/height. ```output-size = in-width * in-height * out_channels```
    * @param has_bias         [in] whether to add a bias vector to the filter outputs
    * @param w_stride         [in] specify the horizontal interval at which to apply the filters to the input
    * @param h_stride         [in] specify the vertical interval at which to apply the filters to the input
    * @param backend_type [in] specify backend engine you use   
    **/
    convolutional_layer(serial_size_t              in_width,
                        serial_size_t              in_height,
                        serial_size_t              window_width,
                        serial_size_t              window_height,
                        serial_size_t              in_channels,
                        serial_size_t              out_channels,
                        const connection_table& connection_table,
                        padding                 pad_type = padding::valid,
                        bool                    has_bias = true,
                        serial_size_t              w_stride = 1,
                        serial_size_t              h_stride = 1,
                        backend_t      backend_type = core::default_engine())
        : Base(std_input_order(has_bias)) {
            conv_set_params(shape3d(in_width, in_height, in_channels),
                            window_width, window_height,
                            out_channels, pad_type, has_bias,
                            w_stride, h_stride,
                            connection_table);
            init_backend(backend_type);
            Base::set_backend_type(backend_type);
    }

    // move constructor
    convolutional_layer(convolutional_layer&& other)  // NOLINT
            : Base(std::move(other))
            , params_(std::move(other.params_))
            , padding_op_(std::move(other.padding_op_))
            , kernel_fwd_(std::move(other.kernel_fwd_))
            , kernel_back_(std::move(other.kernel_back_))
            , cws_(std::move(other.cws_)) {
        init_backend(std::move(other.engine()));
    }

    ///< number of incoming connections for each output unit
    serial_size_t fan_in_size() const override {
        return params_.weight.width_  *
               params_.weight.height_ * params_.in.depth_;
    }

    ///< number of outgoing connections for each input unit
    serial_size_t fan_out_size() const override  {
        return (params_.weight.width_  / params_.w_stride) *
               (params_.weight.height_ / params_.h_stride) *
                params_.out.depth_;
    }

    /**
     * @param in_data      input vectors of this layer (data, weight, bias)
     * @param out_data     output vectors
     **/
    void forward_propagation(const std::vector<tensor_t*>& in_data,
                             std::vector<tensor_t*>&       out_data) override { 
        // apply padding to the input tensor
        padding_op_.copy_and_pad_input(*in_data[0], cws_.prev_out_padded_);

        std::vector<tensor_t*> in_data_(in_data.size());
        in_data_[0] = in_data_padded(in_data);

        for (serial_size_t i = 1; i < in_data.size(); ++i) {
            in_data_[i] = in_data[i];
        }

        // forward convolutional op context
        auto ctx = OpKernelContext(in_data_, out_data);
             ctx.setParallelize(layer::parallelize());
             ctx.setEngine(layer::engine());

        // launch convolutional kernel
        kernel_fwd_->compute(ctx);

        // activations
        // TODO(edgar/nyanp): refactor and move activations outside
        this->forward_activation(*out_data[0], *out_data[1]);
    }

    /**
     * return delta of previous layer (delta=\frac{dE}{da}, a=wx in fully-connected layer)
     * @param in_data      input vectors (same vectors as forward_propagation)
     * @param out_data     output vectors (same vectors as forward_propagation)
     * @param out_grad     gradient of output vectors (i-th vector correspond with out_data[i])
     * @param in_grad      gradient of input vectors (i-th vector correspond with in_data[i])
     **/
    void back_propagation(const std::vector<tensor_t*>& in_data,
                          const std::vector<tensor_t*>& out_data,
                          std::vector<tensor_t*>&       out_grad,
                          std::vector<tensor_t*>&       in_grad) override {
        // activations
        // TODO(edgar/nyanp): refactor and move activations outside
        this->backward_activation(*out_grad[0], *out_data[0], *out_grad[1]);

        std::vector<tensor_t*> in_data_;
        in_data_.push_back(in_data_padded(in_data));

        for (serial_size_t i = 1; i < in_data.size(); ++i) {
            in_data_.push_back(in_data[i]);
        }

        std::vector<tensor_t*> in_grad_;
        for (serial_size_t i = 0; i < in_grad.size(); ++i) {
            in_grad_.push_back(in_grad[i]);
        }

        if (params_.pad_type == padding::same) {
            in_grad_[0] = &cws_.prev_delta_padded_;
        }

        auto ctx = OpKernelContext(in_data_, out_data, out_grad, in_grad_);
             ctx.setParams(&params_);
             ctx.setParallelize(layer::parallelize());
             ctx.setEngine(layer::engine());

        // launch convolutional kernel
        kernel_back_->compute(ctx);

        // unpad deltas
        padding_op_.copy_and_unpad_delta(cws_.prev_delta_padded_, *in_grad[0]);
    }

    void set_sample_count(serial_size_t sample_count) override {
        Base::set_sample_count(sample_count);
        cws_.prev_delta_padded_.resize(
            sample_count,
            vec_t(params_.in_padded.size(), float_t(0)));
    }

    std::vector<index3d<serial_size_t>> in_shape() const override {
        if (params_.has_bias) {
            return { params_.in, params_.weight,
                index3d<serial_size_t>(1, 1, params_.out.depth_) };
        }
        else {
            return { params_.in, params_.weight };
        }
    }

    std::vector<index3d<serial_size_t>>
    out_shape() const override { return { params_.out, params_.out }; }

    std::string layer_type() const override {
        return std::string("conv");
    }

    //TODO(edgar): check this
    std::string kernel_file() const override {
        return std::string("../tiny_cnn/core/kernels/cl_kernels/conv_layer_spatial.cl");
    }

    //TODO(edgar): is it really needed?
    std::string kernel_header() const override {
        std::stringstream ss;
        ss << "#define MULTI\n";
        ss << "#define KERNEL_H " << params_.weight.height_ << "\n";
        ss << "#define KERNEL_W " << params_.weight.width_  << "\n";
        ss << "#define CHANNELS " << params_.weight.depth_  << "\n";
        ss << "#define STRIDE_H " << params_.h_stride << "\n";
        ss << "#define STRIDE_W " << params_.w_stride << "\n";
        ss << "#define APPLY_BIAS " << params_.has_bias   << "\n";
        ss << "#define OUTPUT_Z "   << params_.out.depth_ << "\n";
        // TODO(edgar): REVISE THIS
        ss << "#define ZPAR " << params_.out.depth_  << "\n";
        return ss.str();
    }

    image<> weight_to_image() const {
        image<> img;
        const serial_size_t border_width = 1;
        const auto pitch = params_.weight.width_ + border_width;
        const auto width = params_.out.depth_ * pitch + border_width;
        const auto height = params_.in.depth_ * pitch + border_width;
        const image<>::intensity_t bg_color = 255;
        const vec_t& W = *this->weights()[0];

        img.resize(width, height);
        img.fill(bg_color);

        auto minmax = std::minmax_element(W.begin(), W.end());

        for (serial_size_t r = 0; r < params_.in.depth_; ++r) {
            for (serial_size_t c = 0; c < params_.out.depth_; ++c) {
                if (!params_.tbl.is_connected(c, r)) continue;

                const auto top  = r * pitch + border_width;
                const auto left = c * pitch + border_width;

                serial_size_t idx = 0;

                for (serial_size_t y = 0; y < params_.weight.height_; ++y) {
                    for (serial_size_t x = 0; x < params_.weight.width_; ++x) {
                        idx = c * params_.in.depth_ + r;
                        idx = params_.weight.get_index(x, y, idx);
                        const float_t w = W[idx];

                        img.at(left + x, top + y)
                            = static_cast<image<>::intensity_t>(
                                rescale(w, *minmax.first,
                                        *minmax.second, 0, 255));
                    }
                }
            }
        }
        return img;
    }


    template <class Archive>
    static void load_and_construct(
        Archive & ar, cereal::construct<convolutional_layer> & construct) {
        serial_size_t w_width, w_height, out_ch, w_stride, h_stride;
        bool has_bias;
        shape3d in;
        padding pad_type;
        connection_table tbl;

        ar(cereal::make_nvp("in_size", in),
            cereal::make_nvp("window_width", w_width),
            cereal::make_nvp("window_height", w_height),
            cereal::make_nvp("out_channels", out_ch),
            cereal::make_nvp("connection_table", tbl),
            cereal::make_nvp("pad_type", pad_type),
            cereal::make_nvp("has_bias", has_bias),
            cereal::make_nvp("w_stride", w_stride),
            cereal::make_nvp("h_stride", h_stride)
        );

        construct(in.width_, in.height_, w_width, w_height, in.depth_,
                  out_ch, tbl, pad_type, has_bias, w_stride, h_stride);
    }

    template <class Archive>
    void serialize(Archive & ar) {
        layer::serialize_prolog(ar);
        ar(cereal::make_nvp("in_size", params_.in),
            cereal::make_nvp("window_width", params_.weight.width_),
            cereal::make_nvp("window_height", params_.weight.height_),
            cereal::make_nvp("out_channels", params_.out.depth_),
            cereal::make_nvp("connection_table", params_.tbl),
            cereal::make_nvp("pad_type", params_.pad_type),
            cereal::make_nvp("has_bias", params_.has_bias),
            cereal::make_nvp("w_stride", params_.w_stride),
            cereal::make_nvp("h_stride", params_.h_stride)
            );
    }

private:
    tensor_t* in_data_padded(const std::vector<tensor_t*>& in) {
        return (params_.pad_type == padding::valid) ?
            in[0] : &cws_.prev_out_padded_;
    }

    void conv_set_params(const shape3d& in,
                         serial_size_t     w_width,
                         serial_size_t     w_height,
                         serial_size_t     outc,
                         padding        ptype,
                         bool           has_bias,
                         serial_size_t     w_stride,
                         serial_size_t     h_stride,
                         const connection_table& tbl = connection_table()) {
        params_.in = in;
        params_.in_padded = shape3d(in_length(in.width_, w_width, ptype),
                                    in_length(in.height_, w_height, ptype),
                                    in.depth_);
        params_.out =
            shape3d(conv_out_length(in.width_, w_width, w_stride, ptype),
                    conv_out_length(in.height_, w_height, h_stride, ptype),
                    outc);
        params_.weight   = shape3d(w_width, w_height, in.depth_ * outc);
        params_.has_bias = has_bias;
        params_.pad_type = ptype;
        params_.w_stride = w_stride;
        params_.h_stride = h_stride;
        params_.tbl      = tbl;

        // init padding buffer
        if (params_.pad_type == padding::same) {
            cws_.prev_delta_padded_.resize(1, vec_t(params_.in_padded.size(), float_t(0)));
        }

        // set parameters to padding operation
        padding_op_ = Conv2dPadding(params_);
    }

    serial_size_t in_length(serial_size_t in_length,
                         serial_size_t window_size, padding pad_type) const {
        return pad_type == padding::same ?
               (in_length + window_size - 1) : in_length;
    }

    static serial_size_t conv_out_dim(serial_size_t in_width,
                                   serial_size_t in_height,
                                   serial_size_t window_size,
                                   serial_size_t w_stride,
                                   serial_size_t h_stride, padding pad_type) {
        return conv_out_length(in_width, window_size, w_stride, pad_type) *
               conv_out_length(in_height, window_size, h_stride, pad_type);
    }

    serial_size_t conv_out_dim(serial_size_t in_width,
                            serial_size_t in_height,
                            serial_size_t window_width,
                            serial_size_t window_height,
                            serial_size_t w_stride,
                            serial_size_t h_stride, padding pad_type) const {
        return conv_out_length(in_width, window_width, w_stride, pad_type) *
               conv_out_length(in_height, window_height, h_stride, pad_type);
    }

    void createOp() override {
        init_backend(layer::engine());
    }

    void init_backend(const backend_t backend_type) {
        core::OpKernelConstruction ctx =
        core::OpKernelConstruction(layer::device(), &params_);

        if (backend_type == backend_t::internal ||
            backend_type == backend_t::nnpack   ||
            backend_type == backend_t::avx) {
            
            kernel_fwd_.reset(new Conv2dOp(ctx));
            kernel_back_.reset(new Conv2dGradOp(ctx));
            return;
        }
        else if (backend_type == backend_t::opencl) {
            throw nn_error("Not implemented engine: " + to_string(backend_type));
            /*kernel_fwd_.reset(new Conv2dOpenCLForwardOp(ctx));
            kernel_back_.reset(new Conv2dOpenCLBackwardOp(ctx));
            return;*/
        }
        else if (backend_type == backend_t::libdnn) {
            if (layer::device() == nullptr) return;
            kernel_fwd_.reset(new Conv2dLibDNNForwardOp(ctx));
            kernel_back_.reset(new Conv2dLibDNNBackwardOp(ctx));
            return;
        }
        else {
            throw nn_error("Not supported engine: " + to_string(backend_type));
        }

    }

 private:
    /* The convolution parameters */
    conv_params params_;

    /* Padding operation */
    Conv2dPadding padding_op_;

    /* Forward and backward ops */
    std::shared_ptr<core::OpKernel> kernel_fwd_;
    std::shared_ptr<core::OpKernel> kernel_back_;

    /* Buffer to store padded data */
    struct conv_layer_worker_specific_storage {
        tensor_t prev_out_padded_;
        tensor_t prev_delta_padded_;
    } cws_;
};

}  // namespace tiny_dnn
