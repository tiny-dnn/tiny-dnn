/*
    COPYRIGHT

    All contributions by Taiga Nomi
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.

    All other contributions:
    Copyright (c) 2013-2016, the respective contributors.
    All rights reserved.

    Each contributor holds copyright over their respective contributions.
    The project versioning (Git) records all such contribution source information.

    LICENSE

    The BSD 3-Clause License


    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name of tiny-dnn nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include "tiny_dnn/core/framework/device.fwd.h"
#include "tiny_dnn/core/params/conv_params.h"

namespace tiny_dnn {
namespace core {

class OpKernel;  // delared below

class OpKernelConstruction {
 public:
    explicit OpKernelConstruction() {}
    explicit OpKernelConstruction(Device* device, Params* params)
        : device_(device), params_(params) {}
    
    // Returns the device raw pointer
    Device* device() const { return device_; }

    // Returns the device raw pointer
    Params* params() const { return params_; }

 private:
    Device* device_ = nullptr;
    Params* params_ = nullptr;
};

class OpKernelContext {
 public:
    struct OpParams {
        // the op kernel being computed.
        OpKernel* op_kernel_ptr = nullptr;

        // the device on which the kernel is running.
        Device* device_ptr = nullptr;

        // the layer on which kernel is runnning
        layer* layer_ptr_ = nullptr;

        // the operation params
        Params* params_ptr_ = nullptr;

        // parallelize operation
        bool parallelize = false;

        backend_t engine = default_engine();
    };

    explicit OpKernelContext(const std::vector<tensor_t*>& in_data,
                             std::vector<tensor_t*>&       out_data)
            : in_data_(in_data), out_data_(out_data) {
        op_params_ = std::unique_ptr<OpParams>(new OpParams());
    }

    explicit OpKernelContext(const std::vector<tensor_t*>& in_data,
                             const std::vector<tensor_t*>& out_data,
                             std::vector<tensor_t*>&       out_grad,
                             std::vector<tensor_t*>&       in_grad)
            : in_data_(in_data)
            , out_data_(out_data)
            , out_grad_(out_grad)
            , in_grad_(in_grad) {
        op_params_ = std::unique_ptr<OpParams>(new OpParams());
    }

    tensor_t& input(const int idx) const {
        return *in_data_[idx];
    }

    tensor_t& output(const int idx) const {
        return *out_data_[idx];
    }

    tensor_t& input_grad(const int idx) const {
        return *in_grad_[idx];
    }

    tensor_t& output_grad(const int idx) const {
        return *out_grad_[idx];
    }

    void setParams(Params* params) {
        op_params_->params_ptr_ = params;
    }

    Params* params() const {
        return op_params_->params_ptr_;
    }

    void setParallelize(const bool parallelize) {
        op_params_->parallelize = parallelize;
    }

    bool parallelize() const {
        return op_params_->parallelize;
    }

    void setDevice(Device* device) {
        op_params_->device_ptr = device;
    }

    Device* device() const {
        return op_params_->device_ptr;
    }

    void setLayer(layer* layer) {
        op_params_->layer_ptr_ = layer;
    }

    layer* Layer() const {
        return op_params_->layer_ptr_;
    }

    backend_t engine() const {
        return op_params_->engine;
    }

    void setEngine(const backend_t engine) {
        op_params_->engine = engine;
    }

 private:
    std::vector<tensor_t*> in_data_;
    std::vector<tensor_t*> out_data_;
    std::vector<tensor_t*> out_grad_;
    std::vector<tensor_t*> in_grad_;

    std::unique_ptr<OpParams> op_params_;
};

class OpKernel {
 public:
    explicit OpKernel() {}
    explicit OpKernel(const OpKernelConstruction& context)
        : device_(context.device())
        , params_(context.params()) {}

    virtual ~OpKernel() {}

    virtual void compute(const OpKernelContext& context) = 0;

 protected:
    Device* device_ = nullptr;
    Params* params_ = nullptr;
};

}  // namespace core
}  // namespace tiny_dnn
