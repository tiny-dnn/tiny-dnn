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
#include "tiny_dnn/util/product.h"

namespace tiny_dnn {

/**
 * compute fully-connected(matmul) operation
 **/
template<typename Activation>
class quantized_fully_connected_layer : public feedforward_layer<Activation> {
public:
    typedef feedforward_layer<Activation> Base;
    CNN_USE_LAYER_MEMBERS;

    /**
     * @param in_dim [in] number of elements of the input
     * @param out_dim [in] number of elements of the output
     * @param has_bias [in] whether to include additional bias to the layer
     **/
    quantized_fully_connected_layer(serial_size_t in_dim,
                                    serial_size_t out_dim,
                                    bool       has_bias = true,
                                    backend_t  backend_type = core::backend_t::internal)
            : Base(std_input_order(has_bias)) {
        set_params(in_dim, out_dim, has_bias);
        init_backend(backend_type);
    }

    // move constructor
    quantized_fully_connected_layer(quantized_fully_connected_layer&& other)
            : Base(std::move(other))
            , params_(std::move(other.params_)) {
        init_backend(core::backend_t::internal);
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
                             std::vector<tensor_t*>& out_data) override {
        if (in_data.size() == 2 || in_data.size() == 3) {
            Base::backend_->fully_q(in_data, out_data);

            // activations
            this->forward_activation(*out_data[0], *out_data[1]);
        } else if (in_data.size() == 4 || in_data.size() == 6) {
            Base::backend_->fully_eq(in_data, out_data);
        }
    }

    void back_propagation(const std::vector<tensor_t*>& in_data,
                          const std::vector<tensor_t*>& out_data,
                          std::vector<tensor_t*>&       out_grad,
                          std::vector<tensor_t*>&       in_grad) override {
        Base::backend_->fully_q(in_data, out_data, out_grad, in_grad);
    }

    std::string layer_type() const override { return "q_fully-connected"; }

protected:
    fully_params params_;

    void set_params(const serial_size_t in_size,
                    const serial_size_t out_size,
                    bool             has_bias) {
        params_.in_size_  = in_size;
        params_.out_size_ = out_size;
        params_.has_bias_ = has_bias;
    }

    void init_backend(backend_t backend_type) {
        std::shared_ptr<core::backend> backend = nullptr;

        // allocate new backend
        if (backend_type == backend_t::internal) {
            backend = std::make_shared<core::tiny_backend>(&params_,
                [this](const tensor_t& p_delta,
                       const tensor_t& out, tensor_t& c_delta) {
                     return Base::backward_activation(p_delta, out, c_delta);
                });
        } else if (backend_type == backend_t::nnpack) {
            backend = std::make_shared<core::nnp_backend>(&params_);
        } else if (backend_type == backend_t::libdnn) {
            backend = std::make_shared<core::dnn_backend>();
        } else {
            throw nn_error("Not supported backend type.");
        }

        if (backend) {
            Base::set_backend(backend);
            Base::set_backend_type(backend_type);
            Base::backend_->set_layer(this);
        } else {
            throw nn_error("Could not allocate the backend.");
        }
    }
};

} // namespace tiny_dnn
