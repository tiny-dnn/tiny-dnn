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

#include "tiny_cnn/layers/layer.h"
#include "tiny_cnn/core/params/conv_params.h"
#include "tiny_cnn/core/params/deconv_params.h"
#include "tiny_cnn/core/params/maxpool_params.h"
#include "tiny_cnn/core/params/fully_params.h"

namespace tiny_cnn {
namespace core {

class context;

enum class backend_t { tiny_cnn, nnpack, libdnn, avx, };

struct backend_params {
    backend_params() {}
};

class backend {
 public:
    // context holds solution-dependent parameters
    // context should be able to hold any types of structures (like boost::any)
    explicit backend(context* ctx_ = nullptr) {}

    // core math functions

    virtual void conv2d(cnn_size_t                 index,
                        const std::vector<vec_t*>& in_data,
                        std::vector<vec_t*>&       out_data) = 0;

    virtual void conv2d(cnn_size_t                 index,
                        const std::vector<vec_t*>& in_data,
                        const std::vector<vec_t*>& out_data,
                        std::vector<vec_t*>&       out_grad,
                        std::vector<vec_t*>&       in_grad) = 0;

    virtual void deconv2d(cnn_size_t                 index,
                          const std::vector<vec_t*>& in_data,
                          std::vector<vec_t*>&       out_data) = 0;

    virtual void deconv2d(cnn_size_t                 index,
                               const std::vector<vec_t*>& in_data,
                               const std::vector<vec_t*>& out_data,
                               std::vector<vec_t*>&       out_grad,
                               std::vector<vec_t*>&       in_grad) = 0;

    virtual void matmul() = 0;

    virtual void maxpool(cnn_size_t                 index,
                         const std::vector<vec_t*>& in_data,
                         std::vector<vec_t*>&       out_data) = 0;

    virtual void maxpool(cnn_size_t                 index,
                         const std::vector<vec_t*>& in_data,
                         const std::vector<vec_t*>& out_data,
                         std::vector<vec_t*>&       out_grad,
                         std::vector<vec_t*>&       in_grad) = 0;

    virtual void fully(cnn_size_t                 index,
                       const std::vector<vec_t*>& in_data,
                       std::vector<vec_t*>&       out_data) = 0;

    virtual void fully(cnn_size_t                 index,
                       const std::vector<vec_t*>& in_data,
                       const std::vector<vec_t*>& out_data,
                       std::vector<vec_t*>&       out_grad,
                       std::vector<vec_t*>&       in_grad) = 0;

    context* get_context() const { return ctx_; }

    void set_layer(layerptr_t layer) { layer_ = layer; }
    void set_type(const backend_t type) { type_ = type; }
    backend_t get_type() const { return type_; }

 protected:
    context* ctx_;
    layerptr_t layer_;
    backend_t type_;
};

}  // namespace core
}  // namespace tiny_cnn
