/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once
#include <algorithm>
#include <map>
#include <string>
#include <vector>
#include "tiny_dnn/layers/cell.h"
#include "tiny_dnn/layers/layer.h"

namespace tiny_dnn {

/*
 * optional parameters for the recurrent layer
*/
struct recurrent_layer_parameters {
  // clip gradients to this value
  float_t clip = 0;
  // the backend engine
  core::backend_t backend_type = core::default_engine();
  // max number of sequential steps to remember
  size_t bptt_max = 0;  // 0 will set it to "seq_len"
  // whether to reset the state between timesteps
  bool reset_state = true;
};

/**
 * Recurrent layer
 * ===============
 * Wrapper for recurrent layers, manages the state of a recurrent cells.
 *
 * Receives a tensor of (seq_length * batch_size) elements:
 *
 * ``{seq1_batch1 seq1_batch2, ..., seq2_batch1, seq2_batch2, ...
 * seqn_batchn}``,
 *
 * which is forwarded/backwarded sequence-wise, feeding the output state of each
 * step to the input of the next one.
 *
 * By default, the state is reset every time that a tensor of `seq_len *
 * batch_size` is processed. However, this can be avoided by increasing the max
 * backpropagation-through-time steps, which is the number of iterations until
 * the state is reset. This way sequences can be fed one by one, and the state
 *will
 * be carried through all the process, until `params.bptt_max` iterations
 * have been reached (the default value of `bptt_max` is seq_len, see the
 *constructor).
 *
 * When the state is reset, it is set to zero. However it can be set to copy the
 * state given in the input data by using `reset_state(true)`, or
 *`params.reset_state`
 * in the constructor.
 *
 **/
class recurrent_layer : public layer {
 public:
  /**
   * @param cell [in] pointer to the wrapped cell
   * @param seq_len [in] length of the input sequences
   * @param params [in] recurrent layer optional [parameters] @ref
   *recurrent_layer_parameters "recurrent_layer_parameters"
   **/
  recurrent_layer(
    std::shared_ptr<cell> cell_p,
    size_t seq_len,
    const recurrent_layer_parameters params = recurrent_layer_parameters())
    : layer(cell_p->input_order(), cell_p->output_order()),
      cell_(cell_p),
      clip_(params.clip),
      bptt_max_((params.bptt_max > 0 ? params.bptt_max : seq_len)),
      bptt_count_(0),
      reset_state_(params.reset_state),
      seq_len_(seq_len) {
    layer::set_backend_type(params.backend_type);
    cell_->init_backend(
      static_cast<layer *>(this));  // depends on layer::set_backend_type!
    init();
  }

  // move constructor
  recurrent_layer(recurrent_layer &&other)
    : layer(std::move(other)),
      cell_(std::move(other.cell_)),
      bptt_max_(std::move(other.bptt_max_)),
      bptt_count_(std::move(other.bptt_count_)),
      reset_state_(std::move(other.reset_state_)),
      seq_len_(std::move(other.seq_len_)) {
    cell_->init_backend(static_cast<layer *>(this));
    init();
  }

  ~recurrent_layer() {
    for (size_t i = 0; i < input_buffer_.size(); i++) {
      if (delete_mask_[i]) {
        delete input_buffer_[i];
      }
      delete input_grad_buffer_[i];
    }
    for (size_t o = 0; o < output_buffer_.size(); o++) {
      delete output_buffer_[o];
      delete output_grad_buffer_[o];
    }
  }

  /**
   * Set the max sequence length to remember.
   * @param val [in] max input sequence length for backpropagation through time.
   */
  void bptt_max(size_t val) { bptt_max_ = val; }

  /**
   * Clip gradients to given value. Helps to prevent gradient explosion.
   * @param in  [in] Gradient vector.
   * @param val [in] Value to clamp.
   * @param out [out] Clamped vector.
   */
  inline void clip(const vec_t &in, const float_t val, vec_t &out) {
    for_(layer::parallelize(), 0u, in.size(),
         [&](const blocked_range &range) {
           for (size_t i = range.begin(); i < range.end(); i++) {
             out[i] = std::max<float_t>(-val, std::min<float_t>(in[i], val));
           }
         },
         0u);
  }

  size_t fan_in_size(size_t i) const override {
    return cell_->in_shape()[i].width_;
  }

  size_t fan_out_size(size_t i) const override {
    return cell_->in_shape()[i].height_;
  }

  std::vector<index3d<size_t>> in_shape() const override {
    return cell_->in_shape();
  }

  std::vector<index3d<size_t>> out_shape() const override {
    return cell_->out_shape();
  }

  /**
   * Forward propagation through time. Copies each sequence to a buffer and
   * forwards it propagating the hidden states.
   * @param in_data  [in]  input tensors. Data must be of size (seq_length *
   * batch_size, dim2, ..., dimn).
   * @param out_data [out] output tensors.
   */
  void forward_propagation(const std::vector<tensor_t *> &in_data,
                           std::vector<tensor_t *> &out_data) override {
    size_t batch_size = (*out_data[0]).size() / seq_len_;
    // create buffers to store the batches of the sequences
    reshape_forward_buffers_(batch_size, in_data);

    // truncated backprop through time
    if (bptt_count_ == 0) {
      if (reset_state_) {
        clear_state();
      } else {
        for (size_t i = 0; i < in_type_.size(); i++) {
          if (in_type_[i] == vector_type::aux) {
            *input_buffer_[i] = *in_data[i];
          }
        }
      }
    }

    size_t start = 0;  // auxiliary variable
    for (size_t s = 0; s < seq_len_; s++) {
      start = s * batch_size;
      for (size_t i = 0; i < in_data.size(); i++) {
        // move current sequence batch to a buffer
        if (in_type_[i] == vector_type::data) {
          auto *data       = &(*in_data[i])[start];
          tensor_t &buffer = *input_buffer_[i];
          for (size_t b = 0; b < batch_size; b++) {
            buffer[b] = data[b];  // copy data
          }
        } else if (in_type_[i] == vector_type::aux) {
          auto *data       = &(*in_data[i])[start];
          tensor_t &buffer = *input_buffer_[i];
          if (!reset_state_) {
            for (size_t b = 0; b < batch_size; b++) {
              data[b] = buffer[b];  // copy state
            }
          }
        }
      }
      // forward current sequence batch
      cell_->forward_propagation(input_buffer_, output_buffer_);
      // move from buffer to output
      for (size_t o = 0; o < out_data.size(); o++) {
        tensor_t &out_buffer = *output_buffer_[o];
        auto *data           = &(*out_data[o])[start];
        for (size_t b = 0; b < batch_size; b++) {
          data[b] = out_buffer[b];
        }
        // copy output state to next input
        if (state_mask_[o]) {
          auto &in_buffer = *input_buffer_[state_map_o2i_[o]];
          for (size_t b = 0; b < batch_size; b++) {
            in_buffer[b] = out_buffer[b];
          }
        }
      }
    }
    bptt_count_ = (bptt_count_ + seq_len_) % bptt_max_;
  }

  /**
   * Back propagation through time. Copies each sequence to a buffer and
   * backwards it copying state input grads to
   * previous sequence output grads.
   * @param in_data  [in]  input tensors. Data must be of size (seq_length *
   * batch_size, dim2, ..., dimn).
   * @param out_data [in]  output tensors.
   * @param out_grad [in]  next layer gradients.
   * @param in_grad  [out] computed gradients.
   */
  void back_propagation(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) override {
    size_t batch_size = (*out_data[0]).size() / seq_len_;
    // resize input buffers
    reshape_backward_buffers_(batch_size, in_data);

    // move input to buffer
    for (int s = (seq_len_ - 1); s >= 0; s--) {
      const size_t start = batch_size * s;
      for (size_t i = 0; i < in_data.size(); i++) {
        if (in_type_[i] == vector_type::data ||
            in_type_[i] == vector_type::aux) {
          auto *data       = &(*in_data[i])[start];
          tensor_t &buffer = *input_buffer_[i];
          for (size_t b = 0; b < batch_size; b++) {
            buffer[b] = data[b];
          }
        }
        auto &grad_buffer = *input_grad_buffer_[i];
        fill_tensor(grad_buffer, 0.0);
      }
      // fill output
      for (size_t o = 0; o < out_data.size(); o++) {
        tensor_t &buffer      = *output_buffer_[o];
        tensor_t &grad_buffer = *output_grad_buffer_[o];
        auto *data            = &(*out_data[o])[start];
        auto *grad            = &(*out_grad[o])[start];
        int end               = seq_len_ - 1;
        if (out_type_[o] == vector_type::aux && s == end && reset_state_) {
          fill_tensor(buffer, 0.0);
          fill_tensor(grad_buffer, 0.0);
        } else {
          for (size_t b = 0; b < batch_size; b++) {
            buffer[b]      = data[b];
            grad_buffer[b] = grad[b];
          }
        }
      }
      cell_->back_propagation(input_buffer_, output_buffer_,
                              output_grad_buffer_, input_grad_buffer_);
      for (size_t i = 0; i < in_data.size(); i++) {
        auto *in_grad_   = &(*in_grad[i])[start];
        tensor_t &buffer = *input_grad_buffer_[i];
        for (size_t b = 0; b < batch_size; b++) {
          if (clip_ > 0) {
            clip(buffer[b], clip_, in_grad_[b]);
          } else {
            in_grad_[b] = buffer[b];
          }
        }
      }
      // copy gradient to previous state
      if (s > 0) {
        for (size_t o = 0; o < out_data.size(); o++) {
          if (state_mask_[o]) {
            tensor_t &buffer  = *input_grad_buffer_[state_map_o2i_[o]];
            size_t prev_start = batch_size * (s - 1);
            auto *out_grad_   = &(*out_grad[o])[prev_start];
            for (size_t b = 0; b < batch_size; b++) {
              out_grad_[b] = buffer[b];
            }
          }
        }
      }
    }
  }

  std::string layer_type() const override { return "recurrent-layer"; }

  /**
   * Zeroes the hidden state.
   */
  void clear_state() {
    for (size_t i = 0; i < input_buffer_.size(); i++) {
      if (in_type_[i] == vector_type::aux) {
        fill_tensor(*input_buffer_[i], 0.0);
      }
    }
    bptt_count_ = 0;
  }

  /**
   * Whether to remember the previous state at each forward/backward call.
   * @param reset [in] if false, sequences can be fed element by element (state
   * is remembered).
   * @param clear [in] whether to zero the current state.
   */
  void reset_state(bool reset, bool clear = true) {
    reset_state_ = reset;
    if (clear) clear_state();
  }

  /**
   * Sets the current input sequence length.
   * @param len [in] current input sequence length.
   */
  void seq_len(size_t len) { seq_len_ = len; }

  friend struct serialization_buddy;

 private:
  void init() {
    input_buffer_.resize(in_shape().size());
    input_grad_buffer_.resize(in_shape().size());
    output_buffer_.resize(out_shape().size());
    output_grad_buffer_.resize(out_shape().size());
    delete_mask_.resize(input_buffer_.size(), false);
    std::vector<size_t> state_pos;
    for (size_t i = 0; i < in_shape().size(); i++) {
      input_grad_buffer_[i] = new tensor_t();
      if (in_type_[i] == vector_type::data || in_type_[i] == vector_type::aux) {
        input_buffer_[i] = new tensor_t();
        delete_mask_[i]  = true;
        if (in_type_[i] == vector_type::aux) {
          state_pos.push_back(i);
        }
      }
    }
    for (size_t o = 0; o < out_shape().size(); o++) {
      output_buffer_[o]      = new tensor_t();
      output_grad_buffer_[o] = new tensor_t();
      size_t map_size        = state_map_o2i_.size();
      if (in_type_[o] == vector_type::aux && map_size < state_pos.size()) {
        state_map_o2i_[o] = state_pos[map_size];
        state_mask_.push_back(true);
      } else {
        state_mask_.push_back(false);
      }
    }
  }

  // Helper function to set internal input buffers to the correct size.
  inline void reshape_forward_buffers_(const size_t batch_size,
                                       const std::vector<tensor_t *> &in_data) {
    for (size_t i = 0; i < in_data.size(); i++) {
      auto &buffer   = *input_buffer_[i];
      auto in_shape_ = in_shape();
      // weights and biases do not change with the length of the sequences
      if (in_type_[i] == vector_type::weight ||
          in_type_[i] == vector_type::bias) {
        input_buffer_[i] = in_data[i];
      } else {
        buffer.resize(batch_size);
        if (in_type_[i] == vector_type::aux) {
          for (size_t b = 0; b < batch_size; b++) {
            buffer[b].resize(in_shape_[i].size(), 0);
          }
        }
      }
    }
    auto out_shape_ = out_shape();
    for (size_t o = 0; o < out_shape_.size(); o++) {
      tensor_t &buffer = *output_buffer_[o];
      buffer.resize(batch_size);
      for (size_t b = 0; b < batch_size; b++) {
        buffer[b].resize(out_shape_[o].size(), 0);
      }
    }
  }

  // Helper function to set internal output buffers to the correct size.
  inline void reshape_backward_buffers_(
    const size_t batch_size, const std::vector<tensor_t *> &in_data) {
    for (size_t i = 0; i < in_data.size(); i++) {
      auto &buffer   = *input_buffer_[i];
      auto in_shape_ = in_shape();
      // weights and biases do not change with the length of the sequences
      if (in_type_[i] == vector_type::weight ||
          in_type_[i] == vector_type::bias) {
        input_buffer_[i] = in_data[i];
      } else {
        buffer.resize(batch_size);
      }
      // resize input gradient
      tensor_t &grad_buffer = *input_grad_buffer_[i];
      const size_t in_size  = in_shape()[i].size();
      if (grad_buffer.size() != batch_size) {
        grad_buffer.resize(batch_size);
      }
      for (size_t b = 0; b < batch_size; b++) {
        grad_buffer[b].resize(in_size);
      }
    }
    // resize output
    for (size_t o = 0; o < out_shape().size(); o++) {
      const size_t out_size = out_shape()[o].size();
      tensor_t &buffer      = *output_buffer_[o];
      tensor_t &grad_buffer = *output_grad_buffer_[o];
      buffer.resize(batch_size);
      grad_buffer.resize(batch_size);
      for (size_t b = 0; b < batch_size; b++) {
        buffer[b].resize(out_size);
        grad_buffer[b].resize(out_size);
      }
    }
  }

  // unique pointer to wrapped cell
  std::shared_ptr<cell> cell_;

  float_t clip_;
  size_t bptt_max_;
  size_t bptt_count_;

  bool reset_state_ = true;
  // sequence length
  // TODO(prlz77) std::vector, support variable sequence lengths
  size_t seq_len_;

  std::map<size_t, size_t> state_map_o2i_;
  std::vector<bool> state_mask_;

  // buffers for state transitions
  std::vector<tensor_t *> input_buffer_;
  std::vector<tensor_t *> output_buffer_;
  std::vector<tensor_t *> input_grad_buffer_;
  std::vector<tensor_t *> output_grad_buffer_;
  std::vector<bool> delete_mask_;
};

}  // namespace tiny_dnn
