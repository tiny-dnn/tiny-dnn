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
#include "tiny_dnn/core/kernels/lstm_cell_grad_op.h"
#include "tiny_dnn/core/kernels/lstm_cell_op.h"
#include "tiny_dnn/layers/cell.h"

namespace tiny_dnn {

/*
 *  Optional LSTM cell parameters
 */
struct lstm_cell_parameters {
  // whether the layer uses biases
  bool has_bias = true;
};

/**
 * Long short-term memory cell (LSTM)
 * ==================================
 * Note: Layers finished in `_cell` should not be used directly but wrapped in a
 *layer (see [recurrent_layer](@ref recurrent_layer), [rnn](@ref rnn)).
 *
 * ```
 * i[t] = σ(W[x->i]x[t] + W[h->i]h[t−1] + b[1->i])                      (1)
 * f[t] = σ(W[x->f]x[t] + W[h->f]h[t−1] + b[1->f])                      (2)
 * z[t] = tanh(W[x->c]x[t] + W[h->c]h[t−1] + b[1->c])                   (3)
 * c[t] = f[t]c[t−1] + i[t]z[t]                                         (4)
 * o[t] = σ(W[x->o]x[t] + W[h->o]h[t−1] + b[1->o])                      (5)
 * h[t] = o[t]tanh(c[t])                                                (6)
 * ```
 * References
 * * [Long short-term
 *memory](http://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735)
 * * [Understanding LSTM
 *Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
 * * [ElementResearch](https://github.com/Element-Research/rnn#rnn.FastLSTM)
 **/
class lstm_cell : public cell {
 public:
  /**
   * @param in_dim [in] number of elements of the input
   * @param out_dim [in] number of elements of the output
   * @param params [in] optional lstm cell [parameters](@ref
   *lstm_cell_parameters).
   **/
  lstm_cell(size_t in_dim,
            size_t out_dim,
            const lstm_cell_parameters &params = lstm_cell_parameters()) {
    set_params(in_dim, out_dim, params.has_bias);
  }

  // move constructor
  lstm_cell(lstm_cell &&other)
    : cell(std::move(other)),
      params_(std::move(other.params_)),
      fwd_ctx_(std::move(other.fwd_ctx_)),
      bwd_ctx_(std::move(other.bwd_ctx_)),
      kernel_fwd_(std::move(other.kernel_fwd_)),
      kernel_back_(std::move(other.kernel_back_)) {}

  inline std::vector<vector_type> input_order() {
    std::vector<vector_type> types = {vector_type::data,  // input vector
                                      vector_type::aux,  // input state (h(t-1))
                                      vector_type::aux,  // memory (c(t-1))
                                      vector_type::weight,   // W[x->i]
                                      vector_type::weight,   // W[x->f]
                                      vector_type::weight,   // W[x->c]
                                      vector_type::weight,   // W[x->o]
                                      vector_type::weight,   // W[h->i]
                                      vector_type::weight,   // W[h->f]
                                      vector_type::weight,   // W[h->c]
                                      vector_type::weight};  // W[h->o]
    if (params_.has_bias_) {
      types.push_back(vector_type::bias);  // bi
      types.push_back(vector_type::bias);  // bf
      types.push_back(vector_type::bias);  // bc
      types.push_back(vector_type::bias);  // bo
    }
    return types;
  }

  inline std::vector<vector_type> output_order() {
    return {vector_type::data,  // output vector
            vector_type::aux,   // output state  h(t)
            vector_type::aux,   // output memory c(t)
            vector_type::aux,   // aux state i(t)
            vector_type::aux,   // aux state f(t)
            vector_type::aux,   // aux state z(t)
            vector_type::aux};  // tanh(c(t))
  }

  inline size_t fan_in_size(size_t i) const { return in_shape()[i].width_; }

  inline size_t fan_out_size(size_t i) const { return in_shape()[i].height_; }

  inline std::vector<index3d<size_t>> in_shape() const {
    std::vector<index3d<size_t>> shape = {
      index3d<size_t>(params_.in_size_, 1, 1),                    // x
      index3d<size_t>(params_.out_size_, 1, 1),                   // h(t-1)
      index3d<size_t>(params_.out_size_, 1, 1),                   // c(t-1)
      index3d<size_t>(params_.in_size_, params_.out_size_, 1),    // W[x->i]
      index3d<size_t>(params_.in_size_, params_.out_size_, 1),    // W[x->f]
      index3d<size_t>(params_.in_size_, params_.out_size_, 1),    // W[x->c]
      index3d<size_t>(params_.in_size_, params_.out_size_, 1),    // W[x->o]
      index3d<size_t>(params_.out_size_, params_.out_size_, 1),   // W[h->i]
      index3d<size_t>(params_.out_size_, params_.out_size_, 1),   // W[h->f]
      index3d<size_t>(params_.out_size_, params_.out_size_, 1),   // W[h->c]
      index3d<size_t>(params_.out_size_, params_.out_size_, 1)};  // W[h->o]
    if (params_.has_bias_) {
      shape.push_back(index3d<size_t>(params_.out_size_, 1, 1));  // bi
      shape.push_back(index3d<size_t>(params_.out_size_, 1, 1));  // bf
      shape.push_back(index3d<size_t>(params_.out_size_, 1, 1));  // bc
      shape.push_back(index3d<size_t>(params_.out_size_, 1, 1));  // bo
    }
    return shape;
  }

  inline std::vector<index3d<size_t>> out_shape() const {
    return {index3d<size_t>(params_.out_size_, 1, 1),   // o(t)
            index3d<size_t>(params_.out_size_, 1, 1),   // h(t)
            index3d<size_t>(params_.out_size_, 1, 1),   // c(t)
            index3d<size_t>(params_.out_size_, 1, 1),   // i(t)
            index3d<size_t>(params_.out_size_, 1, 1),   // f(t)
            index3d<size_t>(params_.out_size_, 1, 1),   // z(t)
            index3d<size_t>(params_.out_size_, 1, 1)};  // tanh(c(t))
  }

  inline void forward_propagation(const std::vector<tensor_t *> &in_data,
                                  std::vector<tensor_t *> &out_data) {
    // forward lstm op context
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
    // backward lstm op context
    bwd_ctx_.set_in_out(in_data, out_data, out_grad, in_grad);
    bwd_ctx_.setParallelize(cell::wrapper_->parallelize());
    bwd_ctx_.setEngine(cell::wrapper_->engine());

    // launch recurrent kernel
    kernel_back_->compute(bwd_ctx_);
  }

  inline std::string layer_type() const { return "lstm-cell"; }

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
    kernel_fwd_.reset(new LSTMCellOp(ctx));
    kernel_back_.reset(new LSTMCellGradOp(ctx));
  }

 private:
  /* The layer parameters */
  core::lstm_cell_params params_;

  /* forward op context */
  core::OpKernelContext fwd_ctx_;

  /* backward op context */
  core::OpKernelContext bwd_ctx_;

  /* Forward and backward ops */
  std::shared_ptr<core::OpKernel> kernel_fwd_;
  std::shared_ptr<core::OpKernel> kernel_back_;
};

}  // namespace tiny_dnn
