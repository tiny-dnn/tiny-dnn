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

#ifdef USE_OPENCL
#include "third_party/CLCudaAPI/clpp11.h"
#endif

namespace tiny_cnn {
namespace core {

class dnn_backend : public backend {
 public:
    // context holds solution-dependent parameters
    // context should be able to hold any types of structures (like boost::any)
    dnn_backend() {}

    // convolution
    dnn_backend(conv_params* params,
                std::function<void(const tensor_t&)> f,
                conv_layer_worker_specific_storage* ptr)
        : params_c_(params),
          conv_layer_worker_storage_(ptr),
          copy_and_pad_input(f) {}

    // core math functions

    void conv2d(const std::vector<tensor_t*>& in_data,
                std::vector<tensor_t*>&       out_data) override {
#ifdef CNN_USE_LIBDNN
        copy_and_pad_input(*in_data[0]);
        const vec_t& W    = (*in_data[1])[0];
        const vec_t& bias = (*in_data[2])[0];
        tensor_t&     a    = *out_data[1];
        const std::vector<const vec_t*> &in = (*conv_layer_worker_storage_).prev_out_padded_; // input // NOLINT
 
        fill_tensor(a, float_t(0));

        for (cnn_size_t i = 0; i < in.size(); i++) {
            kernels::libdnn_conv2d_kernel(*params_c_,
                *in[i], W, bias, a[i]);
        }
        // kernels::libdnn_conv2d_kernel(*params_c_, in, W, bias, a);
#else
        throw nn_error("Tiny-cnn has not been compiled with LibDNN support.");
#endif
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
 
    backend_t get_type() const { return backend_t::libdnn; }

 private:

    /* Pointer to the convolution parameters */
    conv_params* params_c_;

    /* Pointer to the convolution workers */
    conv_layer_worker_specific_storage* conv_layer_worker_storage_;

    /* Pointers to parent class functions */
    std::function<void(const tensor_t&)> copy_and_pad_input;
};

}  // namespace core
}  // namespace tiny_cnn
