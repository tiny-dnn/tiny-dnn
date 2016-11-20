/*
    Copyright (c) 2015, Taiga Nomi
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

#include <string>
#include <vector>
#include <algorithm>

#include "tiny_dnn/core/backend_tiny.h"
#include "tiny_dnn/core/backend_nnp.h"
#include "tiny_dnn/core/backend_dnn.h"
#ifdef CNN_USE_AVX
#include "tiny_dnn/core/backend_avx.h"
#endif

#include "tiny_dnn/core/kernels/maxpool_op.h"
#include "tiny_dnn/core/kernels/maxpool_grad_op.h"

#include "tiny_dnn/util/util.h"
#include "tiny_dnn/util/image.h"
#include "tiny_dnn/activations/activation_function.h"

namespace tiny_dnn {

/**
 * applies max-pooing operaton to the spatial data
 **/
template <typename Activation = activation::identity>
class max_pooling_layer : public feedforward_layer<Activation> {
 public:
    CNN_USE_LAYER_MEMBERS;
    typedef feedforward_layer<Activation> Base;

    /**
     * @param in_width     [in] width of input image
     * @param in_height    [in] height of input image
     * @param in_channels  [in] the number of input image channels(depth)
     * @param pooling_size [in] factor by which to downscale
     **/
    max_pooling_layer(serial_size_t in_width,
                      serial_size_t in_height,
                      serial_size_t in_channels,
                      serial_size_t pooling_size,
                      backend_t  backend_type = core::default_engine())
        : max_pooling_layer(in_width, in_height, in_channels, pooling_size,
                            pooling_size, backend_type) {}

    max_pooling_layer(const shape3d& in_shape,
                      serial_size_t     pooling_size,
                      serial_size_t     stride,
                      backend_t      backend_type = core::default_engine())
        : max_pooling_layer(in_shape.width_, in_shape.height_, in_shape.depth_,
                            pooling_size, stride, backend_type) {}

    max_pooling_layer(serial_size_t in_width,
                      serial_size_t in_height,
                      serial_size_t in_channels,
                      serial_size_t pooling_size,
                      serial_size_t stride,
                      backend_t  backend_type = core::default_engine())
        : max_pooling_layer(in_width, in_height, in_channels, pooling_size,
                            pooling_size, stride, stride, padding::valid,
                            backend_type) {}

    /**
     * @param in_width     [in] width of input image
     * @param in_height    [in] height of input image
     * @param in_channels  [in] the number of input image channels(depth)
     * @param pooling_size [in] factor by which to downscale
     * @param stride       [in] interval at which to apply the filters to the input
    **/
    max_pooling_layer(serial_size_t in_width,
                      serial_size_t in_height,
                      serial_size_t in_channels,
                      serial_size_t pooling_size_x,
                      serial_size_t pooling_size_y,
                      serial_size_t stride_x,
                      serial_size_t stride_y,
                      padding    pad_type = padding::valid,
                      backend_t  backend_type = core::default_engine())
            : Base({ vector_type::data }) {
        set_maxpool_params(
            shape3d(in_width, in_height, in_channels),
            shape3d(conv_out_length(in_width, pooling_size_x, stride_x, pad_type),
                    conv_out_length(in_height, pooling_size_y, stride_y, pad_type),
                    in_channels),
            pooling_size_x, pooling_size_y, stride_x, stride_y, pad_type);

        init_connection();
        init_backend(backend_type);
        Base::set_backend_type(backend_type);
    }

    // move constructor
    max_pooling_layer(max_pooling_layer&& other)  // NOLINT
            : Base(std::move(other))
            , params_(std::move(other.params_)) {
        init_connection();
        init_backend(std::move(Base::engine()));
    }

    serial_size_t fan_in_size() const override {
        return static_cast<serial_size_t>(params_.out2in[0].size());
    }

    serial_size_t fan_out_size() const override {
        return 1;
    }

    void forward_propagation(const std::vector<tensor_t*>& in_data,
                             std::vector<tensor_t*>&       out_data) override {
	// forward convolutional op context
        auto ctx = OpKernelContext(in_data, out_data);
             ctx.setParallelize(layer::parallelize());
             ctx.setEngine(layer::engine());

        // launch convolutional kernel
        kernel_fwd_->compute(ctx);

        // activations
        this->forward_activation(*out_data[0], *out_data[1]);
    }

    void back_propagation(const std::vector<tensor_t*>& in_data,
                          const std::vector<tensor_t*>& out_data,
                          std::vector<tensor_t*>&       out_grad,
                          std::vector<tensor_t*>&       in_grad) override {
	// activations
        // TODO(edgar/nyanp): refactor and move activations outside
        this->backward_activation(*out_grad[0], *out_data[0], *out_grad[1]);

        // backward convolutional op context
        auto ctx = OpKernelContext(in_data, out_data, out_grad, in_grad);
             ctx.setParallelize(layer::parallelize());
             ctx.setEngine(layer::engine());

        // launch convolutional kernel
        kernel_back_->compute(ctx);
    }

    std::vector<index3d<serial_size_t>>
    in_shape() const override { return { params_.in }; }

    std::vector<index3d<serial_size_t>>
    out_shape() const override { return { params_.out, params_.out }; }

    std::string layer_type() const override {
        return std::string("max-pool");
    }

    std::string kernel_file() const override {
        return std::string("../tiny_cnn/core/kernels/cl_kernels/pooling.cl");
    }

    std::pair<serial_size_t, serial_size_t> pool_size() const {
	return std::make_pair(params_.pool_size_x, params_.pool_size_y);
    }

    void set_sample_count(serial_size_t sample_count) override {
        Base::set_sample_count(sample_count);
        params_.out2inmax.resize(
	     sample_count, std::vector<serial_size_t>(params_.out.size()));
    }


    template <class Archive>
    static void
    load_and_construct(Archive & ar,
		       cereal::construct<max_pooling_layer> & construct) {
        shape3d in;
        serial_size_t stride_x, stride_y, pool_size_x, pool_size_y;
        padding pad_type;

        ar(cereal::make_nvp("in_size", in),
           cereal::make_nvp("pool_size_x", pool_size_x),
           cereal::make_nvp("pool_size_y", pool_size_y),
           cereal::make_nvp("stride_x", stride_x),
           cereal::make_nvp("stride_y", stride_y),
           cereal::make_nvp("pad_type", pad_type));
        construct(in.width_, in.height_, in.depth_, pool_size_x, pool_size_y,
		  stride_x, stride_y, pad_type);
    }

    template <class Archive>
    void serialize(Archive & ar) {
        layer::serialize_prolog(ar);
        ar(cereal::make_nvp("in_size", params_.in),
            cereal::make_nvp("pool_size_x", params_.pool_size_x),
            cereal::make_nvp("pool_size_y", params_.pool_size_y),
            cereal::make_nvp("stride_x", params_.stride_x),
            cereal::make_nvp("stride_y", params_.stride_y),
            cereal::make_nvp("pad_type", params_.pad_type));
    }

private:
    /* The Max Poling operation params */
    maxpool_params params_;

    /* Forward and backward ops */
    std::shared_ptr<core::OpKernel> kernel_fwd_;
    std::shared_ptr<core::OpKernel> kernel_back_;

    void connect_kernel(serial_size_t pooling_size_x,
                        serial_size_t pooling_size_y,
                        serial_size_t outx,
                        serial_size_t outy,
                        serial_size_t c) {
        serial_size_t dxmax = static_cast<serial_size_t>(
            std::min(static_cast<serial_size_t>(pooling_size_x),
                     params_.in.width_ - outx * params_.stride_x));

        serial_size_t dymax = static_cast<serial_size_t>(
            std::min(static_cast<serial_size_t>(pooling_size_y),
                     params_.in.height_ - outy * params_.stride_y));

        for (serial_size_t dy = 0; dy < dymax; dy++) {
            for (serial_size_t dx = 0; dx < dxmax; dx++) {
                serial_size_t in_index = params_.in.get_index(
                    static_cast<serial_size_t>(outx * params_.stride_x + dx),
                    static_cast<serial_size_t>(outy * params_.stride_y + dy), c);
                serial_size_t out_index = params_.out.get_index(outx, outy, c);

                if (in_index >= params_.in2out.size()) {
                    throw nn_error("index overflow");
                }
                if (out_index >= params_.out2in.size()) {
                    throw nn_error("index overflow");
                }
                params_.in2out[in_index] = out_index;
                params_.out2in[out_index].push_back(in_index);
            }
        }
    }

    void init_connection() {
        params_.in2out.resize(params_.in.size());
        params_.out2in.resize(params_.out.size());

        for (serial_size_t c = 0; c < params_.in.depth_; ++c) {
            for (serial_size_t y = 0; y < params_.out.height_; ++y) {
                for (serial_size_t x = 0; x < params_.out.width_; ++x) {
                    connect_kernel(params_.pool_size_x,
                                   params_.pool_size_y,
                                   x, y, c);
                }
            }
        }
    }

    void init_backend(backend_t backend_type) {
	core::OpKernelConstruction ctx =
        core::OpKernelConstruction(layer::device(), &params_);

        if (backend_type == backend_t::internal ||
	    backend_type == backend_t::nnpack   ||
            backend_type == backend_t::avx) {

            kernel_fwd_.reset(new MaxPoolOp(ctx));
            kernel_back_.reset(new MaxPoolGradOp(ctx));
            return;
        }
        else {
            throw nn_error("Not supported engine: " + to_string(backend_type));
        }

    }

    void set_maxpool_params(const shape3d& in,
                            const shape3d& out,
                            serial_size_t pooling_size_x,
                            serial_size_t pooling_size_y,
                            serial_size_t stride_x,
                            serial_size_t stride_y,
                            padding pad_type) {
        params_.in          = in;
        params_.out         = out;
        params_.pool_size_x = pooling_size_x;
        params_.pool_size_y = pooling_size_y;
        params_.stride_x    = stride_x;
        params_.stride_y    = stride_y;
        params_.pad_type    = pad_type;
    }
};

}  // namespace tiny_dnn

