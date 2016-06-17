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
#include "tiny_cnn/util/util.h"
#include "tiny_cnn/util/image.h"
#include "tiny_cnn/activations/activation_function.h"

namespace tiny_cnn {

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
    max_pooling_layer(cnn_size_t     in_width,
                      cnn_size_t     in_height,
                      cnn_size_t     in_channels,
                      cnn_size_t     pooling_size,
                      backend_t      backend_type = backend_t::tiny_cnn,
                      backend_params b_params = backend_params())
        : Base( { vector_type::data } )
        , pool_size_(pooling_size)
        , stride_(pooling_size)
        , backend_type_(backend_type)
        , in_(in_width, in_height, in_channels)
        , out_(in_width  / pooling_size,
               in_height / pooling_size,
               in_channels) {
        if ((in_width % pooling_size) || (in_height % pooling_size)) {
            pooling_size_mismatch(in_width, in_height, pooling_size);
        }

        init_connection();
        init_backend(backend_type_);
    }

    /**
     * @param in_width     [in] width of input image
     * @param in_height    [in] height of input image
     * @param in_channels  [in] the number of input image channels(depth)
     * @param pooling_size [in] factor by which to downscale
     * @param stride       [in] interval at which to apply the filters to the input
    **/
    max_pooling_layer(cnn_size_t     in_width,
                      cnn_size_t     in_height,
                      cnn_size_t     in_channels,
                      cnn_size_t     pooling_size,
                      cnn_size_t     stride,
                      backend_t      backend_type = backend_t::tiny_cnn,
                      backend_params b_params = backend_params())
        : Base({vector_type::data})
        , pool_size_(pooling_size)
        , stride_(stride)
        , backend_type_(backend_type)
        , in_(in_width, in_height, in_channels)
        , out_(pool_out_dim(in_width, pooling_size, stride),
               pool_out_dim(in_height, pooling_size, stride),
               in_channels) {
        init_connection();
        init_backend(backend_type_);
    }

    // move constructor
    max_pooling_layer(max_pooling_layer&& other)
            : Base(std::move(other))
            , pool_size_(std::move(other.pool_size_))
            , stride_(std::move(other.stride_))
            , out2in_(std::move(other.out2in_))
            , in2out_(std::move(other.in2out_))
            , backend_type_(std::move(other.backend_type_))
            , max_pooling_layer_worker_storage_(
                std::move(other.max_pooling_layer_worker_storage_))
            , in_(std::move(other.in_))
            , out_(std::move(other.out_)) {
        init_connection();
        init_backend(backend_type_);
    }

    size_t fan_in_size() const override {
        return out2in_[0].size();
    }

    size_t fan_out_size() const override {
        return 1;
    }

    void forward_propagation(cnn_size_t index,
                             const std::vector<vec_t*>& in_data,
                             std::vector<vec_t*>&       out_data) {
        // launch maxpool kernel
        Base::backend_->maxpool(index, in_data, out_data);

        // activations
        vec_t& out     = *out_data[0];
        const vec_t& a = *out_data[1];

        for_i(parallelize_, out.size(), [&](int i) {
            out[i] = h_.f(a, i);
        });
    }

    void back_propagation(cnn_size_t                 index,
                          const std::vector<vec_t*>& in_data,
                          const std::vector<vec_t*>& out_data,
                          std::vector<vec_t*>&       out_grad,
                          std::vector<vec_t*>&       in_grad) {
        // launch maxpool kernel
        Base::backend_->maxpool(index, in_data, out_data,
                                out_grad, in_grad);
    }

    /*void back_propagation_2nd(const std::vector<vec_t>& delta_in) override {
        const vec_t& current_delta2 = delta_in[0];
        const vec_t& prev_out = prev_->output(0);
        const activation::function& prev_h = prev_->activation_function();

        max_pooling_layer_worker_specific_storage& mws = max_pooling_layer_worker_storage_[0];

        for (cnn_size_t i = 0; i < in_size_; i++) {
            cnn_size_t outi = in2out_[i];
            prev_delta2_[i] = (mws.out2inmax_[outi] == i) ? current_delta2[outi] * sqr(prev_h.df(prev_out[i])) : float_t(0);
        }
    }*/

    std::vector<index3d<cnn_size_t>>
    in_shape() const override { return {in_}; }

    std::vector<index3d<cnn_size_t>>
    out_shape() const override { return {out_, out_}; }

    std::string layer_type() const override { return "max-pool"; }
    size_t pool_size() const {return pool_size_;}

    virtual void set_worker_count(cnn_size_t worker_count) override {
        Base::set_worker_count(worker_count);
        max_pooling_layer_worker_storage_.resize(worker_count);
        for (max_pooling_layer_worker_specific_storage& mws :
             max_pooling_layer_worker_storage_) {
            mws.out2inmax_.resize(out_.size());
        }
    }

private:
    size_t pool_size_;
    size_t stride_;

    /* mapping out => in (1:N) */
    std::vector<std::vector<cnn_size_t> > out2in_;
    /* mapping in => out (N:1) */
    std::vector<cnn_size_t> in2out_;

    /* The type of backend */
    backend_t backend_type_;

    std::vector<max_pooling_layer_worker_specific_storage> max_pooling_layer_worker_storage_;

    index3d<cnn_size_t> in_;
    index3d<cnn_size_t> out_;

    static cnn_size_t pool_out_dim(cnn_size_t in_size,
                                   cnn_size_t pooling_size,
                                   cnn_size_t stride) {
        return (int) std::ceil(((float_t)in_size - pooling_size) / stride) + 1;
    }

    void connect_kernel(cnn_size_t pooling_size,
                        cnn_size_t outx,
                        cnn_size_t outy,
                        cnn_size_t c) {
        cnn_size_t dxmax = static_cast<cnn_size_t>(
            std::min((size_t)pooling_size, in_.width_ - outx * stride_));

        cnn_size_t dymax = static_cast<cnn_size_t>(
            std::min((size_t)pooling_size, in_.height_ - outy * stride_));

        for (cnn_size_t dy = 0; dy < dymax; dy++) {
            for (cnn_size_t dx = 0; dx < dxmax; dx++) {
                cnn_size_t in_index = in_.get_index(
                    static_cast<cnn_size_t>(outx * stride_ + dx),
                    static_cast<cnn_size_t>(outy * stride_ + dy), c);
                cnn_size_t out_index = out_.get_index(outx, outy, c);

                if (in_index >= in2out_.size()) {
                    throw nn_error("index overflow");
                }
                if (out_index >= out2in_.size()) {
                    throw nn_error("index overflow");
                }
                in2out_[in_index] = out_index;
                out2in_[out_index].push_back(in_index);
            }
        }
    }

    void init_connection() {
        in2out_.resize(in_.size());
        out2in_.resize(out_.size());

        for (max_pooling_layer_worker_specific_storage& mws :
             max_pooling_layer_worker_storage_) {
            mws.out2inmax_.resize(out_.size());
        }

        for (cnn_size_t c = 0; c < in_.depth_; ++c) {
            for (cnn_size_t y = 0; y < out_.height_; ++y) {
                for (cnn_size_t x = 0; x < out_.width_; ++x) {
                    connect_kernel(static_cast<cnn_size_t>(pool_size_),
                                   x, y, c);
                }
            }
        }
    }

    void init_backend(backend_t backend_type) {
        switch (backend_type) {
            case backend_t::tiny_cnn:
                Base::backend_ = std::make_shared<core::tiny_backend>(
                    &out2in_,
                    &in2out_,
                    [this](const vec_t& p_delta,
                           const vec_t& out, vec_t& c_delta) {
                        return Base::backward_activation(p_delta, out, c_delta);
                    },
                    &max_pooling_layer_worker_storage_);
                Base::backend_->set_layer(this);
                break;
            case backend_t::nnpack:
                Base::backend_ = std::make_shared<core::nnp_backend>();
                Base::backend_->set_layer(this);
                break;
            case backend_t::libdnn:
                Base::backend_ = std::make_shared<core::dnn_backend>();
                Base::backend_->set_layer(this);
                break;
            default:
                throw nn_error("not supported backend type");
        }
    }

};

} // namespace tiny_cnn
