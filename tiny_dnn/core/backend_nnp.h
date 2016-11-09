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
#include "tiny_dnn/core/kernels/nnp_deconv2d_kernel.h"

namespace tiny_dnn {
namespace core {

class nnp_backend : public backend {
 public:
    // context holds solution-dependent parameters
    // context should be able to hold any types of structures (like boost::any)

    // convolution
    nnp_backend(conv_params* params,
                std::function<void(const tensor_t&)> f1,
                conv_layer_worker_specific_storage* ptr)
        : params_c_(params)
        , conv_layer_worker_storage_(ptr)
        , copy_and_pad_input(f1) { init_nnp_engine(); }

    // deconvolution
    explicit nnp_backend(deconv_params* params)
        : params_d_(params) { init_nnp_engine(); }

    // maxpool
    explicit nnp_backend(maxpool_params* params)
        : params_m_(params) { init_nnp_engine(); }

    // fully_connected
    explicit nnp_backend(fully_params* params)
        : params_f_(params) { init_nnp_engine(); }

    nnp_backend() { init_nnp_engine(); }

    // core math functions

    void conv2d(const std::vector<tensor_t*>& in_data,
                std::vector<tensor_t*>&       out_data) override {
	if (params_c_) return;  // workaround to fix warnings
	if (params_f_) return;  // workaround to fix warnings
	if (params_d_) return;  // workaround to fix warnings
	if (conv_layer_worker_storage_) return;    // workaround to fix warnings
	if (deconv_layer_worker_storage_) return;  // workaround to fix warnings

        /*if (!params_c_->has_bias) {
            throw nn_error("NNPACK Convolution requires a bias term.");
        }

        if (params_c_->w_stride != 1 || params_c_->h_stride != 1) {
            throw nn_error("NNPACK Convolution requires stride 1.");
        }

        copy_and_pad_input(*in_data[0]);
        const vec_t& W = (*in_data[1])[0];
        const vec_t& bias = (*in_data[2])[0];
        tensor_t&    a = *out_data[1];
        const std::vector<const vec_t*> &in = (*conv_layer_worker_storage_).prev_out_padded_; // input // NOLINT

        fill_tensor(a, float_t(0));

        // TODO
        throw nn_not_implemented_error();

        kernels::nnp_conv2d_kernel(*params_c_, in, W, bias, a);*/
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
        throw nn_error("NNPACK does not support back propagation.");
    }

    void conv2d_q(const std::vector<tensor_t*>& in_data,
                  const std::vector<tensor_t*>& out_data,
                  std::vector<tensor_t*>&       out_grad,
                  std::vector<tensor_t*>&       in_grad) override {
        throw nn_error("NNPACK does not support back propagation.");
    }

    void deconv2d(const std::vector<tensor_t*>& in_data,
                  std::vector<tensor_t*>&       out_data) override {
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
        throw nn_error("NNPACK does not support back propagation.");
    }

    void deconv2d_q(const std::vector<tensor_t*>& in_data,
                    const std::vector<tensor_t*>& out_data,
                    std::vector<tensor_t*>&       out_grad,
                    std::vector<tensor_t*>&       in_grad) override {
        throw nn_error("NNPACK does not support back propagation.");
    }

    void maxpool(const std::vector<tensor_t*>& in_data,
                 std::vector<tensor_t*>&       out_data) override {
        // just to fix warning: remove in future
        if (params_m_) {}

        /**if (params_m_->stride_x != 2 || params_m_->stride_y != 2) {
            throw nn_error("NNPACK Max-Pool requires a stride == 2.");
        }

        if (params_m_->pool_size_x != 2 || params_m_->pool_size_y != 2) {
            throw nn_error("NNPACK Max-Pool requires a pool size == 2.");
        }

        const tensor_t& in = *in_data[0];
        tensor_t&       a = *out_data[1];

        kernels::nnp_maxpool_kernel(*params_m_, in, a);*/
    }

    void maxpool(const std::vector<tensor_t*>& in_data,
                 const std::vector<tensor_t*>& out_data,
                 std::vector<tensor_t*>&       out_grad,
                 std::vector<tensor_t*>&       in_grad) override {
        throw nn_error("NNPACK does not support back propagation.");
    }

    void fully(const std::vector<tensor_t*>& in_data,
               std::vector<tensor_t*>&       out_data) override {
        /*const tensor_t& in = *in_data[0];
        const vec_t&    W = (*in_data[1])[0];
        vec_t&          b = (*in_data[2])[0];
        tensor_t&       a = *out_data[1];

        kernels::nnp_fully_connected_kernel(*params_f_,
            in, W, b, a, layer_->parallelize());*/
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
        throw nn_error("NNPACK does not support back propagation.");
    }

    void fully_q(const std::vector<tensor_t*>& in_data,
                 const std::vector<tensor_t*>& out_data,
                 std::vector<tensor_t*>&       out_grad,
                 std::vector<tensor_t*>&       in_grad) override {
        throw nn_error("NNPACK does not support back propagation.");
    }

   backend_t type() const override { return backend_t::nnpack; }

 private:
    /* Pointer to the convolution parameters */
    conv_params* params_c_;
    deconv_params* params_d_;
    maxpool_params* params_m_;
    fully_params* params_f_;

    /* Pointer to the convolution workers */
    conv_layer_worker_specific_storage* conv_layer_worker_storage_;
    deconv_layer_worker_specific_storage* deconv_layer_worker_storage_;

    /* Pointers to parent class functions */
    std::function<void(const tensor_t&)> copy_and_pad_input;
    std::function<void(const tensor_t&, tensor_t&)> copy_and_pad_delta;

    void init_nnp_engine() {
#ifdef CNN_USE_NNPACK
        nnp_status init_status = nnp_initialize();
        check_nnp_status(init_status);

        if (init_status != nnp_status_success) {
            throw nn_error("Could not initialize NNPACK.");
        }
#else
        throw nn_error("Tiny-cnn has not been compiled with NNPACK support.");
#endif
    }

#ifdef CNN_USE_NNPACK
    void check_nnp_status(nnp_status status) {
        switch (status) {
            case nnp_status_success:
                break;
            case nnp_status_invalid_batch_size:
                nn_warn("NNPACK function was called with batch_size == 0");
                break;
            case nnp_status_invalid_channels:
                nn_warn("NNPACK function was called with channels == 0.");
                break;
            case nnp_status_invalid_input_channels:
                nn_warn("NNPACK function was called with input_channels == 0.");
                break;
            case nnp_status_invalid_output_channels:
                nn_warn("NNPACK function was called with output_channels == 0.");
                break;
            case nnp_status_invalid_input_size:
                nn_warn(" NNPACK function was called with input_size.height == 0 or input_size.width == 0.");
                break;
            case nnp_status_invalid_input_stride:
                nn_warn(" NNPACK function was called with input_stride.height == 0 or input_stride.width == 0.");
                break;
            case nnp_status_invalid_input_padding:
                nn_warn("NNPACK function was called with input_padding not less than respective kernel (or pooling) size.");
                break;
            case nnp_status_invalid_kernel_size:
                nn_warn("NNPACK function was called with kernel_size.height == 0 or kernel_size.width == 0.");
                break;
            case nnp_status_invalid_pooling_size:
                nn_warn("NNPACK function was called with pooling_size.height == 0 or pooling_size.width == 0.");
                break;
            //case nnp_status_invalid_pooling_stride:
            //    nn_warn("NNPACK function was called with pooling_stride.height == 0 or pooling_stride.width == 0.");
            //    break;
            case nnp_status_invalid_algorithm:
                nn_warn("NNPACK function was called with convolution algorithm not in nnp_convolution_algorithm enumeration.");
                break;
            case nnp_status_unsupported_input_size:
                nn_warn("NNPACK does not support the particular input size for the function.");
                break;
            case nnp_status_unsupported_input_stride:
                nn_warn("NNPACK does not support the particular input sride for the function.");
                break;
            case nnp_status_unsupported_input_padding:
                nn_warn("NNPACK does not support the particular input padding for the function.");
                break;
            case nnp_status_unsupported_kernel_size:
                nn_warn("NNPACK does not support the particular kernel size for the function.");
                break;
            case nnp_status_unsupported_pooling_size:
                nn_warn("NNPACK does not support the particular pooling size for the function.");
                break;
            case nnp_status_unsupported_pooling_stride:
                nn_warn("NNPACK does not support the particular pooling stride for the function .");
                break;
            case nnp_status_unsupported_algorithm:
                nn_warn("NNPACK does not support the particular convolution algorithm for the function.");
                break;
            case nnp_status_uninitialized:
                nn_warn("NNPACK function was called before the library was initialized.");
                break;
            case nnp_status_unsupported_hardware:
                nn_warn("NNPACK does not implement this function for the host CPU.");
                break;
            case nnp_status_out_of_memory:
                nn_warn("NNPACK failed to allocate memory for temporary buffers.");
                break;
        }
    }
#endif
};

}  // namespace core
}  // namespace tiny_dnn
