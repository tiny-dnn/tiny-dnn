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
#include "tiny_dnn/layers/layer.h"

#include "tiny_dnn/core/kernels/fully_connected_op.h"
#include "tiny_dnn/core/kernels/fully_connected_grad_op.h"

namespace tiny_dnn {

/**
 * compute fully-connected(matmul) operation
 **/
template<typename Activation>
class fully_connected_layer : public feedforward_layer<Activation> {
public:
    typedef feedforward_layer<Activation> Base;
    CNN_USE_LAYER_MEMBERS;

    /**
     * @param in_dim [in] number of elements of the input
     * @param out_dim [in] number of elements of the output
     * @param has_bias [in] whether to include additional bias to the layer
     **/
    fully_connected_layer(serial_size_t in_dim,
                          serial_size_t out_dim,
                          bool       has_bias = true,
                          backend_t  backend_type = core::default_engine())
            : Base(std_input_order(has_bias)) {
        set_params(in_dim, out_dim, has_bias);
        init_backend(backend_type);
        Base::set_backend_type(backend_type);
    }

    // move constructor
    fully_connected_layer(fully_connected_layer&& other)
            : Base(std::move(other))
            , params_(std::move(other.params_))
            , kernel_fwd_(std::move(other.kernel_fwd_))
            , kernel_back_(std::move(other.kernel_back_)) {
        init_backend(std::move(other.engine()));
    }

    serial_size_t fan_in_size() const override {
        return params_.in_size_;
    }

    serial_size_t fan_out_size() const override {
        return params_.out_size_;
    }

    std::vector<index3d<serial_size_t>> in_shape() const override {
        if (params_.has_bias_) {
            return { index3d<serial_size_t>(params_.in_size_, 1, 1),
                     index3d<serial_size_t>(params_.in_size_,
                                         params_.out_size_, 1),
                     index3d<serial_size_t>(params_.out_size_, 1, 1) };
        } else {
            return { index3d<serial_size_t>(params_.in_size_, 1, 1),
                     index3d<serial_size_t>(params_.in_size_,
                                         params_.out_size_, 1) };
        }
    }

    std::vector<index3d<serial_size_t>> out_shape() const override {
        return { index3d<serial_size_t>(params_.out_size_, 1, 1),
                 index3d<serial_size_t>(params_.out_size_, 1, 1) };
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

    std::string layer_type() const override { return "fully-connected"; }

    template <class Archive>
    static void load_and_construct(Archive & ar, cereal::construct<fully_connected_layer> & construct) {
        serial_size_t in_dim, out_dim;
        bool has_bias;

        ar(cereal::make_nvp("in_size", in_dim),
           cereal::make_nvp("out_size", out_dim),
           cereal::make_nvp("has_bias", has_bias));
        construct(in_dim, out_dim, has_bias);
    }

    template <class Archive>
    void serialize(Archive & ar) {
        layer::serialize_prolog(ar);
        ar(cereal::make_nvp("in_size", params_.in_size_),
           cereal::make_nvp("out_size", params_.out_size_),
           cereal::make_nvp("has_bias", params_.has_bias_));
    }

protected:

    void set_params(const serial_size_t in_size,
                    const serial_size_t out_size,
                    bool             has_bias) {
        params_.in_size_  = in_size;
        params_.out_size_ = out_size;
        params_.has_bias_ = has_bias;
    }

    void init_backend(backend_t backend_type) {
        core::OpKernelConstruction ctx =
        core::OpKernelConstruction(layer::device(), &params_);

        if (backend_type == backend_t::internal ||
            backend_type == backend_t::avx||
            backend_type == backend_t::nnpack
                ) {

            kernel_fwd_.reset(new FullyConnectedOp(ctx));
            kernel_back_.reset(new FullyConnectedGradOp(ctx));

            return;
        }
        else {
            throw nn_error("Not supported engine: " + to_string(backend_type));
        }
    }

 private:
    /* The layer parameters */
    fully_params params_;

    /* Forward and backward ops */
    std::shared_ptr<core::OpKernel> kernel_fwd_;
    std::shared_ptr<core::OpKernel> kernel_back_;
};

} // namespace tiny_dnn
