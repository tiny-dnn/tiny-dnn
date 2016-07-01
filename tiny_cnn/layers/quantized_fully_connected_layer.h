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
#include "tiny_cnn/layers/layer.h"
#include "tiny_cnn/util/product.h"

namespace tiny_cnn {

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
    quantized_fully_connected_layer(cnn_size_t     in_dim,
                          cnn_size_t     out_dim,
                          bool           has_bias = true,
                          backend_t      backend_type = backend_t::tiny_cnn,
                          backend_params b_params = backend_params())
            : Base(std_input_order(has_bias)) {
        set_params(in_dim, out_dim, has_bias);
        init_backend(backend_type);
    }

    // move constructor
    quantized_fully_connected_layer(quantized_fully_connected_layer&& other)
            : Base(std::move(other))
            , params_(std::move(other.params_)) {
        init_backend(std::move(Base::get_backend_type()));
    }

    size_t fan_in_size() const override {
        return params_.in_size_;
    }

    size_t fan_out_size() const override {
        return params_.out_size_;
    }

    std::vector<index3d<cnn_size_t>> in_shape() const override {
        if (params_.has_bias_) {
            return { index3d<cnn_size_t>(params_.in_size_, 1, 1),
                     index3d<cnn_size_t>(params_.in_size_,
                                         params_.out_size_, 1),
                     index3d<cnn_size_t>(params_.out_size_, 1, 1) };
        } else {
            return { index3d<cnn_size_t>(params_.in_size_, 1, 1),
                     index3d<cnn_size_t>(params_.in_size_,
                                         params_.out_size_, 1) };
        }
    }

    std::vector<index3d<cnn_size_t>> out_shape() const override {
        return { index3d<cnn_size_t>(params_.out_size_, 1, 1),
                 index3d<cnn_size_t>(params_.out_size_, 1, 1) };
    }

    void forward_propagation(cnn_size_t index,
                             const std::vector<vec_t*>& in_data,
                             std::vector<vec_t*>& out_data) override {
        Base::backend_->q_fully(index, in_data, out_data);

        // activations
        vec_t& out     = *out_data[0];
        const vec_t& a = *out_data[1];

        for_i(parallelize_, params_.out_size_, [&](int i) {
            out[i] = this->h_.f(a, i);
        });
    }

    void back_propagation(cnn_size_t                index,
                          const std::vector<vec_t*>& in_data,
                          const std::vector<vec_t*>& out_data,
                          std::vector<vec_t*>&       out_grad,
                          std::vector<vec_t*>&       in_grad) override {
        Base::backend_->fully(index, in_data,
                              out_data, out_grad, in_grad);
    }

    std::string layer_type() const override { return "q_fully-connected"; }

protected:
    fully_params params_;

    void set_params(const cnn_size_t in_size,
                    const cnn_size_t out_size,
                    bool             has_bias) {
        params_.in_size_  = in_size;
        params_.out_size_ = out_size;
        params_.has_bias_ = has_bias;
    }

    void init_backend(backend_t backend_type) {
        std::shared_ptr<core::backend> backend = nullptr;

        // allocate new backend
        if (backend_type == backend_t::tiny_cnn) {
            backend = std::make_shared<core::tiny_backend>(&params_,
                [this](const vec_t& p_delta,
                       const vec_t& out, vec_t& c_delta) {
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
            Base::backend_->set_layer(this);
            Base::backend_->set_type(backend_type);
        } else {
            throw nn_error("Could not allocate the backend.");
        }
    }
};

} // namespace tiny_cnn
