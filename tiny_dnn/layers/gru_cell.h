/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once
#include <string>
#include <vector>
#include "tiny_dnn/activations/sigmoid_layer.h"
#include "tiny_dnn/activations/tanh_layer.h"
#include "tiny_dnn/core/kernels/gru_cell_grad_op.h"
#include "tiny_dnn/core/kernels/gru_cell_op.h"
#include "tiny_dnn/layers/cell.h"

namespace tiny_dnn {

/*
 *  Optional gru cell parameters
 */
struct gru_cell_parameters {
  // whether the layer uses biases
  bool has_bias = true;
};

/**
 * Gated Recurrent Unit (GRU)
 * ==========================
 * Note: Layers finished in `_cell` should not be used directly but wrapped in a
 *layer (see [recurrent_layer](@ref recurrent_layer), [rnn](@ref rnn)).
 * ```
 * z[t] = σ(W[x->z]x[t] + W[s->z]s[t−1] + b[1->z])            (1)
 * r[t] = σ(W[x->r]x[t] + W[s->r]s[t−1] + b[1->r])            (2)
 * h[t] = tanh(W[x->h]x[t] + W[hr->c](s[t−1]r[t]) + b[1->h])  (3)
 * s[t] = (1-z[t])h[t] + z[t]s[t-1]                           (4)
 * ```
 *
 * References
 * * [A Theoretically Grounded Application of Dropout in Recurrent Neural
 *Networks](https://arxiv.org/abs/1512.05287)
 * * https://github.com/Element-Research/rnn#rnn.GRU
 * * http://colah.github.io/posts/2015-08-Understanding-LSTMs/
 *
 **/
class gru_cell : public cell {
 public:
  /**
   * @param in_dim [in] number of elements of the input
   * @param out_dim [in] number of elements of the output
   * @param params [in] optional gru cell [parameters](@ref gru_cell_parameters)
   **/
  gru_cell(size_t in_dim,
           size_t out_dim,
           const gru_cell_parameters &params = gru_cell_parameters()) {
    set_params(in_dim, out_dim, params.has_bias);
  }

  // move constructor
  gru_cell(gru_cell &&other)
    : cell(std::move(other)),
      params_(std::move(other.params_)),
      fwd_ctx_(std::move(other.fwd_ctx_)),
      bwd_ctx_(std::move(other.bwd_ctx_)),
      kernel_fwd_(std::move(other.kernel_fwd_)),
      kernel_back_(std::move(other.kernel_back_)) {}

  inline std::vector<vector_type> input_order() {
    std::vector<vector_type> types = {vector_type::data,     // input vector
                                      vector_type::aux,      // h(t-1)
                                      vector_type::weight,   // W[x->z]
                                      vector_type::weight,   // W[x->r]
                                      vector_type::weight,   // W[x->h]
                                      vector_type::weight,   // W[hr->c]
                                      vector_type::weight,   // W[s->z]
                                      vector_type::weight};  // W[s->r]
    if (params_.has_bias_) {
      types.push_back(vector_type::bias);  // bz
      types.push_back(vector_type::bias);  // br
      types.push_back(vector_type::bias);  // bh
    }
    return types;
  }

  inline std::vector<vector_type> output_order() {
    return {vector_type::data,  // output vector   s(t)
            vector_type::aux,   // output state    s(t)
            vector_type::aux,   // internal state  h(t)
            vector_type::aux,   // reset gate    r(t)
            vector_type::aux,   // update gate   z(t)
            vector_type::aux,   // aux state  hr(t)
            vector_type::aux};  // aux state  - z(t) (1-z)
  }

  inline size_t fan_in_size(size_t i) const { return in_shape()[i].width_; }

  inline size_t fan_out_size(size_t i) const { return in_shape()[i].height_; }

  inline std::vector<index3d<size_t>> in_shape() const {
    std::vector<index3d<size_t>> shape = {
      index3d<size_t>(params_.in_size_, 1, 1),                    // x[t]
      index3d<size_t>(params_.out_size_, 1, 1),                   // s[t-1]
      index3d<size_t>(params_.in_size_, params_.out_size_, 1),    // W[x->x]
      index3d<size_t>(params_.in_size_, params_.out_size_, 1),    // W[x->r]
      index3d<size_t>(params_.in_size_, params_.out_size_, 1),    // W[x->h]
      index3d<size_t>(params_.out_size_, params_.out_size_, 1),   // W[hr->c]
      index3d<size_t>(params_.out_size_, params_.out_size_, 1),   // W[s->z]
      index3d<size_t>(params_.out_size_, params_.out_size_, 1)};  // W[s->r]
    if (params_.has_bias_) {
      shape.push_back(index3d<size_t>(params_.out_size_, 1, 1));  // bz
      shape.push_back(index3d<size_t>(params_.out_size_, 1, 1));  // br
      shape.push_back(index3d<size_t>(params_.out_size_, 1, 1));  // bh
    }
    return shape;
  }

  inline std::vector<index3d<size_t>> out_shape() const {
    return {
      index3d<size_t>(params_.out_size_, 1, 1),   // output vector  s(t)
      index3d<size_t>(params_.out_size_, 1, 1),   // output state   s(t)
      index3d<size_t>(params_.out_size_, 1, 1),   // internal state h(t)
      index3d<size_t>(params_.out_size_, 1, 1),   // reset gate     r(t)
      index3d<size_t>(params_.out_size_, 1, 1),   // update gate    z(t)
      index3d<size_t>(params_.out_size_, 1, 1),   // aux state  hr(t)
      index3d<size_t>(params_.out_size_, 1, 1)};  // aux state  - z(t) (1-z)
  }

  inline void forward_propagation(const std::vector<tensor_t *> &in_data,
                                  std::vector<tensor_t *> &out_data) {
    // forward gru op context
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
    // backward gru op context
    bwd_ctx_.set_in_out(in_data, out_data, out_grad, in_grad);
    bwd_ctx_.setParallelize(cell::wrapper_->parallelize());
    bwd_ctx_.setEngine(cell::wrapper_->engine());

    // launch recurrent kernel
    kernel_back_->compute(bwd_ctx_);
  }

  inline std::string layer_type() const { return "gru-cell"; }

  friend struct serialization_buddy;

 protected:
  void set_params(const size_t in_size, const size_t out_size, bool has_bias) {
    params_.in_size_  = in_size;
    params_.out_size_ = out_size;
    params_.has_bias_ = has_bias;
    params_.tanh_     = std::make_shared<tanh_layer>(tanh_layer());
    params_.sigmoid_  = std::make_shared<sigmoid_layer>(sigmoid_layer());
  }

  void init_backend(const layer *wrapper) {
    cell::set_wrapper(wrapper);
    CNN_UNREFERENCED_PARAMETER(cell::wrapper_->engine());
    core::OpKernelConstruction ctx =
      core::OpKernelConstruction(cell::wrapper_->device(), &params_);
    kernel_fwd_.reset(new GRUCellOp(ctx));
    kernel_back_.reset(new GRUCellGradOp(ctx));
  }

 private:
  /* The layer parameters */
  core::gru_cell_params params_;

  /* forward op context */
  core::OpKernelContext fwd_ctx_;

  /* backward op context */
  core::OpKernelContext bwd_ctx_;

  /* Forward and backward ops */
  std::shared_ptr<core::OpKernel> kernel_fwd_;
  std::shared_ptr<core::OpKernel> kernel_back_;
};

}  // namespace tiny_dnn
