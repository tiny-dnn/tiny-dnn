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

#include "tiny_dnn/layers/layer.h"
#include "tiny_dnn/core/params/conv_params.h"
#include "tiny_dnn/core/params/deconv_params.h"
#include "tiny_dnn/core/params/maxpool_params.h"
#include "tiny_dnn/core/params/fully_params.h"

namespace tiny_dnn {
namespace core {

// TODO(edgar): remove this
class context;

enum class backend_t { internal, nnpack, libdnn, avx, opencl };

inline std::ostream& operator << (std::ostream& os, backend_t type) {
    switch (type) {
        case backend_t::internal: os << "Internal"; break;
        case backend_t::nnpack:   os << "NNPACK";   break;
        case backend_t::libdnn:   os << "LibDNN";   break;
        case backend_t::avx:      os << "AVX";      break;
        case backend_t::opencl:   os << "OpenCL";   break;
        default:
            throw nn_error("Not supported ostream enum.");
            break;
    }
    return os;
}

/*enum class Engine { OpenCL };*/

inline backend_t default_engine() {
#ifdef CNN_USE_AVX
#if defined(__AVX__) || defined(__AVX2__)
    return backend_t::avx;
#endif
#endif // CNN_USE_AVX
    return backend_t::internal;
}

class backend {
 public:
    // context holds solution-dependent parameters
    // context should be able to hold any types of structures (like boost::any)
    explicit backend(context* ctx_ = nullptr) {}

    // core math functions

    virtual void conv2d(const std::vector<tensor_t*>& in_data,
                        std::vector<tensor_t*>&       out_data) = 0;

    virtual void conv2d_q(const std::vector<tensor_t*>& in_data,
                          std::vector<tensor_t*>&       out_data) = 0;

    virtual void conv2d_eq(const std::vector<tensor_t*>& in_data,
                           std::vector<tensor_t*>&       out_data) = 0;

    virtual void conv2d(const std::vector<tensor_t*>& in_data,
                        const std::vector<tensor_t*>& out_data,
                        std::vector<tensor_t*>&       out_grad,
                        std::vector<tensor_t*>&       in_grad) = 0;

    virtual void conv2d_q(const std::vector<tensor_t*>& in_data,
                          const std::vector<tensor_t*>& out_data,
                          std::vector<tensor_t*>&       out_grad,
                          std::vector<tensor_t*>&       in_grad) = 0;

    virtual void deconv2d(const std::vector<tensor_t*>& in_data,
                          std::vector<tensor_t*>&       out_data) = 0;

    virtual void deconv2d_q(const std::vector<tensor_t*>& in_data,
                            std::vector<tensor_t*>&       out_data) = 0;

    virtual void deconv2d_eq(const std::vector<tensor_t*>& in_data,
                             std::vector<tensor_t*>&       out_data) = 0;

    virtual void deconv2d(const std::vector<tensor_t*>& in_data,
                          const std::vector<tensor_t*>& out_data,
                          std::vector<tensor_t*>&       out_grad,
                          std::vector<tensor_t*>&       in_grad) = 0;

    virtual void deconv2d_q(const std::vector<tensor_t*>& in_data,
                            const std::vector<tensor_t*>& out_data,
                            std::vector<tensor_t*>&       out_grad,
                            std::vector<tensor_t*>&       in_grad) = 0;

    virtual void maxpool(const std::vector<tensor_t*>& in_data,
                         std::vector<tensor_t*>&       out_data) = 0;

    virtual void maxpool(const std::vector<tensor_t*>& in_data,
                         const std::vector<tensor_t*>& out_data,
                         std::vector<tensor_t*>&       out_grad,
                         std::vector<tensor_t*>&       in_grad) = 0;

    virtual void fully(const std::vector<tensor_t*>& in_data,
                       std::vector<tensor_t*>&       out_data) = 0;

    virtual void fully_q(const std::vector<tensor_t*>& in_data,
                         std::vector<tensor_t*>&       out_data) = 0;

    virtual void fully_eq(const std::vector<tensor_t*>& in_data,
                          std::vector<tensor_t*>&       out_data) = 0;

    virtual void fully(const std::vector<tensor_t*>& in_data,
                       const std::vector<tensor_t*>& out_data,
                       std::vector<tensor_t*>&       out_grad,
                       std::vector<tensor_t*>&       in_grad) = 0;

    virtual void fully_q(const std::vector<tensor_t*>& in_data,
                         const std::vector<tensor_t*>& out_data,
                         std::vector<tensor_t*>&       out_grad,
                         std::vector<tensor_t*>&       in_grad) = 0;

    context* get_context() const { return ctx_; }

    void set_layer(layerptr_t layer) { layer_ = layer; }

    virtual backend_t type() const = 0;

 protected:
    context* ctx_;
    layerptr_t layer_;
};

}  // namespace core
}  // namespace tiny_dnn
