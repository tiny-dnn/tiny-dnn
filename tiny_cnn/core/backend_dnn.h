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

#include "tiny_cnn/core/backend.h"
#include "tiny_cnn/core/kernels/libdnn_conv2d_kernel.h"

namespace tiny_cnn {
namespace core {

class dnn_backend : public backend {
 public:
    // context holds solution-dependent parameters
    // context should be able to hold any types of structures (like boost::any)
    dnn_backend(conv_params* params,
                std::function<void(const vec_t&, int)> f1,
                std::vector<conv_layer_worker_specific_storage>* ptr)
        : params_(params),
          conv_layer_worker_storage_(ptr),
          copy_and_pad_input(f1) {}

    // core math functions

    void conv2d(cnn_size_t                 index,
                const std::vector<vec_t*>& in_data,
                std::vector<vec_t*>&       out_data) {
#ifdef CNN_USE_LIBDNN
        copy_and_pad_input(*in_data[0], static_cast<int>(index));
        const vec_t& W    = *in_data[1];
        const vec_t& bias = *in_data[2];
        vec_t&       a    = *out_data[1];
        const vec_t &in   = *((*conv_layer_worker_storage_)[index].prev_out_padded_); // input // NOLINT

        std::fill(a.begin(), a.end(), float_t(0));

        kernels::libdnn_conv2d_kernel(*params_, in, W, bias, a);
#else
    throw nn_error("Tiny-cnn has not been compiled with LibDNN support.");
#endif
    }
    
    void conv2d(cnn_size_t                 index,
                const std::vector<vec_t*>& in_data,
                const std::vector<vec_t*>& out_data,
                std::vector<vec_t*>&       out_grad,
                std::vector<vec_t*>&       in_grad) {
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
        throw nn_error("not implemented yet.");
    }

    void conv2d_q(const std::vector<tensor_t*>& in_data,
                  const std::vector<tensor_t*>& out_data,
                  std::vector<tensor_t*>&       out_grad,
                  std::vector<tensor_t*>&       in_grad) override {
        throw nn_error("not implemented yet.");
    }

    void deconv2d(const std::vector<tensor_t*>& in_data,
                  std::vector<tensor_t*>&       out_data) override {
        throw nn_error("not implemented yet.");
    }

    void deconv2d_q(const std::vector<tensor_t*>& in_data,
                    std::vector<tensor_t*>&       out_data) override {
        throw nn_error("not implemented yet.");
    }

    void deconv2d_eq(const std::vector<tensor_t*>& in_data,
                     std::vector<tensor_t*>&       out_data) override {
        throw nn_error("not implemented yet.");
    }

    void deconv2d(const std::vector<tensor_t*>& in_data,
                  const std::vector<tensor_t*>& out_data,
                  std::vector<tensor_t*>&       out_grad,
                  std::vector<tensor_t*>&       in_grad) override {
        throw nn_error("not implemented yet.");
    }

<<<<<<< HEAD
    void deconv2d_q(const std::vector<tensor_t*>& in_data,
                    const std::vector<tensor_t*>& out_data,
                    std::vector<tensor_t*>&       out_grad,
                    std::vector<tensor_t*>&       in_grad) override {
        throw nn_error("not implemented yet.");
    }

    void maxpool(const std::vector<tensor_t*>& in_data,
                 std::vector<tensor_t*>&       out_data) override {
        throw nn_error("not implemented yet.");
    }

    void maxpool(const std::vector<tensor_t*>& in_data,
                 const std::vector<tensor_t*>& out_data,
                 std::vector<tensor_t*>&       out_grad,
                 std::vector<tensor_t*>&       in_grad) override {
        throw nn_error("not implemented yet.");
    }

    void fully(const std::vector<tensor_t*>& in_data,
               std::vector<tensor_t*>&       out_data) override {
        throw nn_error("not implemented yet.");
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
        throw nn_error("not implemented yet.");
    }

    void fully_q(const std::vector<tensor_t*>& in_data,
                 const std::vector<tensor_t*>& out_data,
                 std::vector<tensor_t*>&       out_grad,
                 std::vector<tensor_t*>&       in_grad) override {
        throw nn_error("not implemented yet.");
    }

    backend_t get_type() const override { return backend_t::libdnn; }
=======
    backend_t get_type() const { return backend_t::libdnn; }

 private:

    /* Pointer to the convolution parameters */
    conv_params* params_;

    /* Pointer to the convolution workers */
    std::vector<conv_layer_worker_specific_storage>* conv_layer_worker_storage_;

    /* Pointers to parent class functions */
    std::function<void(const vec_t&, int)> copy_and_pad_input;
>>>>>>> add libdnn kernel
};

}  // namespace core
}  // namespace tiny_cnn
