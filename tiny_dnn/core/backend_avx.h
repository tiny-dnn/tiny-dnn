/*
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

#include "tiny_dnn/core/backend.h"

#include "tiny_dnn/core/kernels/avx_deconv2d_kernel.h"
#include "tiny_dnn/core/kernels/avx_deconv2d_back_kernel.h"

namespace tiny_dnn {
namespace core {

class avx_backend : public backend {
 public:
    // context holds solution-dependent parameters
    // context should be able to hold any types of structures (like boost::any)

    // convolution
    avx_backend(conv_params* params,
                std::function<void(const tensor_t&)> f1,
                std::function<void(const tensor_t&, tensor_t&)> f2,
                std::function<void(const tensor_t&, const tensor_t&, tensor_t&)> f3,
                conv_layer_worker_specific_storage* ptr)
      : params_c_(params)
      , conv_layer_worker_storage_(ptr)
      , copy_and_pad_input(f1)
      , copy_and_unpad_delta(f2)
      , backward_activation(f3) {}

    // deconvolution
    avx_backend(deconv_params* params,
                std::function<void(const tensor_t&)> f1,
                std::function<void(const tensor_t&, tensor_t&)> f2,
                std::function<void(const tensor_t&, const tensor_t&, tensor_t&)> f3,
                deconv_layer_worker_specific_storage* ptr)
      : params_d_(params)
      , deconv_layer_worker_storage_(ptr)
      , copy_and_unpad_output(f1)
      , copy_and_pad_delta(f2)
      , backward_activation(f3) {}

    // maxpooling
    avx_backend(std::vector<std::vector<serial_size_t>>* out2in,
                std::vector<serial_size_t>* in2out,
                std::function<void(const tensor_t&, const tensor_t&, tensor_t&)> f,
                max_pooling_layer_worker_specific_storage* ptr)
      : max_pooling_layer_worker_storage_(ptr)
      , out2in_(out2in)
      , in2out_(in2out)
      , backward_activation(f) {}

    // fully_connected
    avx_backend(fully_params* params,
                std::function<void(const tensor_t&, const tensor_t&, tensor_t&)> f)
      : params_f_(params)
      , backward_activation(f) {}

    // core math functions

    void conv2d(const std::vector<tensor_t*>& in_data,
                std::vector<tensor_t*>&       out_data) override {
	
	if (params_c_) return;  // workaround to fix warnings
	if (params_f_) return;  // workaround to fix warnings
	if (conv_layer_worker_storage_) return;  // workaround to fix warnings
        /*copy_and_pad_input(*in_data[0]);
        const vec_t& W    = (*in_data[1])[0];
        const vec_t& bias = (*in_data[2])[0];
        tensor_t&    a    = *out_data[1];
        const std::vector<const vec_t*> &in = (*conv_layer_worker_storage_).prev_out_padded_; // input // NOLINT

        fill_tensor(a, float_t(0));

        kernels::avx_conv2d_kernel(*params_c_,
            in, W, bias, a, layer_->parallelize());*/
    }

    void conv2d_q(const std::vector<tensor_t*>& in_data,
                  std::vector<tensor_t*>&       out_data) override {
        throw nn_error("not implemented yet.");
    }

    void conv2d_eq(const std::vector<tensor_t*>& in_data,
                   std::vector<tensor_t*>&       out_data) override {
        throw nn_error("not implemented yet.");
    }

    void conv2d(const std::vector<tensor_t*>& in_data,
                const std::vector<tensor_t*>& out_data,
                std::vector<tensor_t*>&       out_grad,
                std::vector<tensor_t*>&       in_grad) override {
        /*conv_layer_worker_specific_storage& cws = (*conv_layer_worker_storage_);

        //std::vector<const vec_t*>& prev_out = cws.prev_out_padded_;
        const vec_t& W  = (*in_data[1])[0];
        tensor_t&    dW = *in_grad[1];
        tensor_t&    db = *in_grad[2];
        tensor_t&    curr_delta = *out_grad[1];
        tensor_t*    prev_delta = (params_c_->pad_type == padding::same) ?
                                   &cws.prev_delta_padded_ : in_grad[0];

        assert(W.size() == params_c_->weight.size());
        assert(dW[0].size() == params_c_->weight.size());
        assert(curr_delta[0].size() ==  layer_->out_shape()[0].size());

        backward_activation(*out_grad[0], *out_data[0], curr_delta);

        fill_tensor(*prev_delta, float_t(0));

        kernels::avx_conv2d_back_kernel(*params_c_,
            prev_out, W, dW, db, curr_delta, prev_delta);

        if (params_c_->pad_type == padding::same) {
            copy_and_unpad_delta(cws.prev_delta_padded_, *in_grad[0]);
        }*/
    }

    void conv2d_q(const std::vector<tensor_t*>& in_data,
                  const std::vector<tensor_t*>& out_data,
                  std::vector<tensor_t*>&       out_grad,
                  std::vector<tensor_t*>&       in_grad) override {
        throw nn_error("not implemented yet.");
    }

    void deconv2d(const std::vector<tensor_t*>&  in_data,
                  std::vector<tensor_t*>&        out_data) override {
        (*deconv_layer_worker_storage_).prev_out_ = in_data[0];
        const vec_t& W = (*in_data[1])[0];
        const vec_t& bias = (*in_data[2])[0];
        tensor_t&       a = *out_data[1];
        const tensor_t &in = *in_data[0]; // input

        fill_tensor(a, float_t(0));

        kernels::avx_deconv2d_kernel(*params_d_,
            in, W, bias, a, layer_->parallelize());

        copy_and_unpad_output(a);
        a = *(*deconv_layer_worker_storage_).curr_out_unpadded_;
    }

    void deconv2d_q(const std::vector<tensor_t*>&  in_data,
                    std::vector<tensor_t*>&        out_data) override {
        throw nn_error("not implemented yet.");
    }

    void deconv2d_eq(const std::vector<tensor_t*>&  in_data,
                     std::vector<tensor_t*>&        out_data) override {
        throw nn_error("not implemented yet.");
    }

    void deconv2d(const std::vector<tensor_t*>& in_data,
                  const std::vector<tensor_t*>& out_data,
                  std::vector<tensor_t*>&       out_grad,
                  std::vector<tensor_t*>&       in_grad) override {

        deconv_layer_worker_specific_storage& cws = (*deconv_layer_worker_storage_);
        if (params_d_->pad_type == padding::same)
            copy_and_pad_delta(cws.curr_delta_padded, *in_grad[0]);

        const tensor_t& prev_out = *(cws.prev_out_);
        const vec_t& W = (*in_data[1])[0];
        tensor_t&    dW = *in_grad[1];
        tensor_t&    db = *in_grad[2];
        tensor_t&    curr_delta = (params_d_->pad_type == padding::same) ? cws.curr_delta_padded : *out_grad[1];
        tensor_t*    prev_delta = in_grad[0];

        assert(W.size() == params_d_->weight.size());
        assert(dW[0].size() == params_d_->weight.size());
        assert(curr_delta[0].size() ==  layer_->out_shape()[0].size());

        backward_activation(*out_grad[0], *out_data[0], curr_delta);

        fill_tensor(*prev_delta, float_t(0));

        kernels::avx_deconv2d_back_kernel(*params_d_,
            prev_out, W, dW, db, curr_delta, prev_delta);
    }

    void deconv2d_q(const std::vector<tensor_t*>& in_data,
                    const std::vector<tensor_t*>& out_data,
                    std::vector<tensor_t*>&       out_grad,
                    std::vector<tensor_t*>&       in_grad) override {
        throw nn_error("not implemented yet.");
    }

    void maxpool(const std::vector<tensor_t*>& in_data,
                 std::vector<tensor_t*>&       out_data) override {
        // just to fix warning. Remove in a future
        if (max_pooling_layer_worker_storage_) {}
        if (out2in_) {}
        if (in2out_) {}

        /*const tensor_t& in  = *in_data[0];
        tensor_t&       a   = *out_data[1];
        std::vector<std::vector<serial_size_t>>& max_idx =
            (*max_pooling_layer_worker_storage_).out2inmax_;

        kernels::avx_maxpool_kernel(in, a,
            max_idx, *out2in_, layer_->parallelize());*/
    }

    void maxpool(const std::vector<tensor_t*>& in_data,
                 const std::vector<tensor_t*>& out_data,
                 std::vector<tensor_t*>&       out_grad,
                 std::vector<tensor_t*>&       in_grad) override {
        /*tensor_t&       prev_delta = *in_grad[0];
        tensor_t&       curr_delta = *out_grad[1];
        std::vector<std::vector<serial_size_t>>& max_idx =
            (*max_pooling_layer_worker_storage_).out2inmax_;

        CNN_UNREFERENCED_PARAMETER(in_data);

        backward_activation(*out_grad[0], *out_data[0], curr_delta);

        kernels::avx_maxpool_back_kernel(prev_delta, curr_delta,
            max_idx, *in2out_,  layer_->parallelize());*/
    }

    void fully(const std::vector<tensor_t*>& in_data,
               std::vector<tensor_t*>&       out_data) override {
        /*const tensor_t& in = *in_data[0];
        const vec_t&    W = (*in_data[1])[0];
        tensor_t&       a = *out_data[1];

        kernels::avx_fully_connected_kernel(*params_f_,
            in, W, params_f_->has_bias_ ? (*in_data[2])[0] : vec_t(),
            a, layer_->parallelize());*/
    }

    void fully_q(const std::vector<tensor_t*>& in_data,
                 std::vector<tensor_t*>&       out_data) override {
        throw nn_error("not implemented yet.");
    }

    void fully_eq(const std::vector<tensor_t*>& in_data,
                  std::vector<tensor_t*>&       out_data) override {
        throw nn_error("not implemented yet.");
    }

    void fully(const std::vector<tensor_t*>& in_data,
               const std::vector<tensor_t*>& out_data,
               std::vector<tensor_t*>&       out_grad,
               std::vector<tensor_t*>&       in_grad) override {
        /*const tensor_t& prev_out = *in_data[0];
        const vec_t&    W = (*in_data[1])[0];
        tensor_t&       dW = *in_grad[1];
        tensor_t&       db = *in_grad[2];
        tensor_t&       prev_delta = *in_grad[0];
        tensor_t&       curr_delta = *out_grad[1];

        backward_activation(*out_grad[0], *out_data[0], curr_delta);

        kernels::avx_fully_connected_back_kernel(*params_f_, prev_out,
            W, dW, prev_delta, curr_delta, db, layer_->parallelize());*/
    }

    void fully_q(const std::vector<tensor_t*>& in_data,
                 const std::vector<tensor_t*>& out_data,
                 std::vector<tensor_t*>&       out_grad,
                 std::vector<tensor_t*>&       in_grad) override {
        throw nn_error("not implemented yet.");
    }

    backend_t type() const override { return backend_t::avx; }

 private:
    /* Pointer to the convolution parameters */
    conv_params* params_c_;
    deconv_params* params_d_;
    fully_params* params_f_;

    /* Pointer to the workers */
    conv_layer_worker_specific_storage* conv_layer_worker_storage_;
    deconv_layer_worker_specific_storage* deconv_layer_worker_storage_;
    max_pooling_layer_worker_specific_storage* max_pooling_layer_worker_storage_;
    std::vector<std::vector<serial_size_t>>* out2in_;
    std::vector<serial_size_t>* in2out_;

    /* Pointers to parent class functions */
    std::function<void(const tensor_t&)> copy_and_pad_input;
    std::function<void(const tensor_t&)> copy_and_unpad_output;
    std::function<void(const tensor_t&, tensor_t&)> copy_and_unpad_delta;
    std::function<void(const tensor_t&, tensor_t&)> copy_and_pad_delta;
    std::function<void(const tensor_t&, const tensor_t&, tensor_t&)> backward_activation;
};

}  // namespace core
}  // namespace tiny_dnn
