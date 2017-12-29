/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once
#include <string>
#include <vector>
#include "tiny_dnn/activations/tanh_layer.h"
#include "tiny_dnn/core/kernels/rnn_cell_grad_op.h"
#include "tiny_dnn/core/kernels/rnn_cell_op.h"
#include "tiny_dnn/layers/cell.h"

namespace tiny_dnn {

/*
 * rnn_cell configurable optional parameters
 */
struct rnn_cell_parameters {
  // whether the cell has bias
  bool has_bias = true;
};

/**
 * Basic RNN cell
 * ==============
 * Note: Layers finished in `_cell` should not be used directly but wrapped in a
 *layer (see [recurrent_layer](@ref recurrent_layer), [rnn](@ref rnn)).
 *
 * Computes the following operations:
 * ```
 * y(t-1)    y(t)
 *   ^        ^
 *   |V+c     | V+c
 * h(t-1) -> h(t)
 *   ^ +b W   ^ +b
 *   |U       |U
 * x(t-1)    x(t)
 *
 * h(t) = tanh(b + W*h(t-1) + U*x(t)) (1)
 * y(t) = c + V*h(t)                  (2)
 * ```
 * See http://www.deeplearningbook.org/contents/rnn.html for details.
 *
 **/
class rnn_cell : public cell {
 public:
  /**
   * @param in_dim [in] number of elements of the input
   * @param out_dim [in] number of elements of the output
   * @param params [in] optional rnn_cell [parameters](@ref
   *rnn_cell_parameters).
   **/
  rnn_cell(size_t in_dim,
           size_t out_dim,
           const rnn_cell_parameters params = rnn_cell_parameters()) {
    set_params(in_dim, out_dim, params.has_bias);
  }

  // move constructor
  rnn_cell(rnn_cell &&other)
    : cell(std::move(other)),
      params_(std::move(other.params_)),
      fwd_ctx_(std::move(other.fwd_ctx_)),
      bwd_ctx_(std::move(other.bwd_ctx_)),
      kernel_fwd_(std::move(other.kernel_fwd_)),
      kernel_back_(std::move(other.kernel_back_)) {}

  inline std::vector<vector_type> input_order() {
    std::vector<vector_type> types = {
      vector_type::data,     // input vector
      vector_type::aux,      // input state (h(t-1))
      vector_type::weight,   // input weights (U)
      vector_type::weight,   // transition weights (W)
      vector_type::weight};  // output weights (V)
    if (params_.has_bias_) {
      types.push_back(vector_type::bias);  // bias
      types.push_back(vector_type::bias);  // rnn bias
    }
    return types;
  }

  inline std::vector<vector_type> output_order() {
    return {vector_type::data,  // output vector
            vector_type::aux};  // output state (h(t))
  }

  inline size_t fan_in_size(size_t i) const { return in_shape()[i].width_; }

  inline size_t fan_out_size(size_t i) const { return in_shape()[i].height_; }

  inline std::vector<index3d<size_t>> in_shape() const {
    std::vector<index3d<size_t>> shape = {
      index3d<size_t>(params_.in_size_, 1, 1),                    // x
      index3d<size_t>(params_.out_size_, 1, 1),                   // h(t-1)
      index3d<size_t>(params_.in_size_, params_.out_size_, 1),    // U
      index3d<size_t>(params_.out_size_, params_.out_size_, 1),   // W
      index3d<size_t>(params_.out_size_, params_.out_size_, 1)};  // V
    if (params_.has_bias_) {
      shape.push_back(index3d<size_t>(params_.out_size_, 1, 1));  // b
      shape.push_back(index3d<size_t>(params_.out_size_, 1, 1));  // c
    }
    return shape;
  }

  inline std::vector<index3d<size_t>> out_shape() const {
    return {index3d<size_t>(params_.out_size_, 1, 1),
            index3d<size_t>(params_.out_size_, 1, 1)};  // h(t)
  }

  inline void forward_propagation(const std::vector<tensor_t *> &in_data,
                                  std::vector<tensor_t *> &out_data) {
    // forward rnn op context
    fwd_ctx_.set_in_out(in_data, out_data);
    fwd_ctx_.setParallelize(cell::wrapper_->parallelize());
    fwd_ctx_.setEngine(cell::wrapper_->engine());

    // launch recurrent kernel
    kernel_fwd_->compute(fwd_ctx_);
  }

  inline void back_propagation(const std::vector<tensor_t *> &in_data,
                               const std::vector<tensor_t *> &out_data,
                               std::vector<tensor_t *> &out_grad,
                               std::vector<tensor_t *> &in_grad) {
    // backward rnn op context
    bwd_ctx_.set_in_out(in_data, out_data, out_grad, in_grad);
    bwd_ctx_.setParallelize(cell::wrapper_->parallelize());
    bwd_ctx_.setEngine(cell::wrapper_->engine());

    // launch recurrent kernel
    kernel_back_->compute(bwd_ctx_);
  }

  inline void set_activation(std::shared_ptr<activation_layer> activation) {
    params_.activation_ = activation;
  }

  inline std::string layer_type() const { return "rnn-cell"; }

  friend struct serialization_buddy;

 protected:
  void set_params(const size_t in_size, const size_t out_size, bool has_bias) {
    params_.in_size_    = in_size;
    params_.out_size_   = out_size;
    params_.has_bias_   = has_bias;
    params_.activation_ = std::make_unique<tanh_layer>(tanh_layer());
  }

  void init_backend(const layer *wrapper) {
    cell::set_wrapper(wrapper);
    CNN_UNREFERENCED_PARAMETER(cell::wrapper_->engine());
    core::OpKernelConstruction ctx =
      core::OpKernelConstruction(cell::wrapper_->device(), &params_);
    kernel_fwd_.reset(new RecurrentCellOp(ctx));
    kernel_back_.reset(new RecurrentCellGradOp(ctx));
  }

 private:
  /* The layer parameters */
  core::rnn_cell_params params_;

  /* forward op context */
  core::OpKernelContext fwd_ctx_;

  /* backward op context */
  core::OpKernelContext bwd_ctx_;

  /* Forward and backward ops */
  std::shared_ptr<core::OpKernel> kernel_fwd_;
  std::shared_ptr<core::OpKernel> kernel_back_;
};

}  // namespace tiny_dnn
