/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "tiny_dnn/core/framework/device.fwd.h"
#include "tiny_dnn/core/params/conv_params.h"
#include "tiny_dnn/parameter.h"

namespace tiny_dnn {
namespace core {

class OpKernel;  // delared below

class OpKernelConstruction {
 public:
  OpKernelConstruction() {}
  explicit OpKernelConstruction(Device *device, Params *params)
    : device_(device), params_(params) {}

  // Returns the device raw pointer
  Device *device() const { return device_; }

  // Returns the device raw pointer
  Params *params() const { return params_; }

 private:
  Device *device_ = nullptr;
  Params *params_ = nullptr;
};

class OpKernelContext {
 public:
  struct OpParams {
    // the op kernel being computed.
    OpKernel *op_kernel_ptr = nullptr;

    // the device on which the kernel is running.
    Device *device_ptr = nullptr;

    // the layer on which kernel is runnning
    layer *layer_ptr_ = nullptr;

    // the operation params
    Params *params_ptr_ = nullptr;

    // the operation params
    Parameters parameters_;

    // parallelize operation
    bool parallelize = false;

    backend_t engine = default_engine();
  };

  OpKernelContext()
    : in_data_(nullptr),
      out_data_(nullptr),
      out_grad_(nullptr),
      in_grad_(nullptr) {
    op_params_ = std::unique_ptr<OpParams>(new OpParams());
  }

  void set_in_out(const std::vector<Tensor<> *> &in_data,
                  std::vector<Tensor<> *> &out_data) {
    in_data_  = const_cast<std::vector<Tensor<> *> *>(&in_data);
    out_data_ = &out_data;
  }

  void set_in_out(const std::vector<Tensor<> *> &in_data,
                  const std::vector<Tensor<> *> &out_data,
                  std::vector<Tensor<> *> &out_grad,
                  std::vector<Tensor<> *> &in_grad) {
    in_data_  = const_cast<std::vector<Tensor<> *> *>(&in_data);
    out_data_ = const_cast<std::vector<Tensor<> *> *>(&out_data);
    out_grad_ = &out_grad;
    in_grad_  = &in_grad;
  }

  Tensor<> &input(const int idx) { return *(*in_data_)[idx]; }
  const Tensor<> &input(const int idx) const { return *(*in_data_)[idx]; }

  Tensor<> &output(const int idx) { return *(*out_data_)[idx]; }
  const Tensor<> &output(const int idx) const { return *(*out_data_)[idx]; }

  Tensor<> &input_grad(const int idx) { return *(*in_grad_)[idx]; }
  const Tensor<> &input_grad(const int idx) const { return *(*in_grad_)[idx]; }

  Tensor<> &output_grad(const int idx) { return *(*out_grad_)[idx]; }
  const Tensor<> &output_grad(const int idx) const {
    return *(*out_grad_)[idx];
  }

  Parameter *ith_parameter(const size_t idx) {
    return op_params_->parameters_.at(idx);
  }

  const Parameter *ith_parameter(const size_t idx) const {
    return op_params_->parameters_.at(idx);
  }

  void setParameters(const Parameters &parameters) {
    op_params_->parameters_ = parameters;
  }

  void setParams(Params *params) { op_params_->params_ptr_ = params; }

  Params *params() const { return op_params_->params_ptr_; }

  void setParallelize(const bool parallelize) {
    op_params_->parallelize = parallelize;
  }

  bool parallelize() const { return op_params_->parallelize; }

  void setDevice(Device *device) { op_params_->device_ptr = device; }

  Device *device() const { return op_params_->device_ptr; }

  void setLayer(layer *layer) { op_params_->layer_ptr_ = layer; }

  layer *Layer() const { return op_params_->layer_ptr_; }

  backend_t engine() const { return op_params_->engine; }

  void setEngine(const backend_t engine) { op_params_->engine = engine; }

 private:
  std::vector<Tensor<> *> *in_data_;
  std::vector<Tensor<> *> *out_data_;
  std::vector<Tensor<> *> *out_grad_;
  std::vector<Tensor<> *> *in_grad_;

  std::unique_ptr<OpParams> op_params_;
};

class OpKernel {
 public:
  OpKernel() {}
  explicit OpKernel(const OpKernelConstruction &context)
    : device_(context.device()), params_(context.params()) {}

  virtual ~OpKernel() {}

  virtual void compute(OpKernelContext &context) = 0;

 protected:
  Device *device_ = nullptr;
  Params *params_ = nullptr;
};

}  // namespace core
}  // namespace tiny_dnn
