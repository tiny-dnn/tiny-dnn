/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/params/recurrent_cell_params.h"

namespace tiny_dnn {
namespace kernels {

template <typename S1,
          typename S2,
          typename S3,
          typename S4,
          typename S5,
          typename S6,
          typename S7,
          typename S8,
          typename S9>
inline void recurrent_cell_op_internal(
  const Tensor<float_t, S1> &in_data,
  const Tensor<float_t, S2> &prev_h,
  const Tensor<float_t, S3> &U,
  const Tensor<float_t, S4> &W,
  const Tensor<float_t, S5> &V,
  const Tensor<float_t, S6> &bias,
  const Tensor<float_t, S7> &c,
  Tensor<float_t, S8> &out_data,
  Tensor<float_t, S9> &out_h,
  const core::recurrent_cell_params &params,
  const bool layer_parallelize) {
  for_i(layer_parallelize, in_data.shape()[0], [&](size_t sample) {
    const auto in = in_data.subView(TensorSingleIndex(sample), TensorAll());
    const auto prev_state =
      prev_h.subView(TensorSingleIndex(sample), TensorAll());
    auto out        = out_data.subView(TensorSingleIndex(sample), TensorAll());
    auto next_state = out_h.subView(TensorSingleIndex(sample), TensorAll());

    for (size_t o = 0; o < params.out_size_; o++) {
      float_t next_state_ = 0;
      // W * h(t-1)
      for (size_t o_2 = 0; o_2 < params.out_size_; o_2++) {
        next_state_ +=
          W.host_at(o_2 * params.out_size_ + o) * prev_state.host_at(o_2);
      }

      // U*x(t)
      for (size_t i = 0; i < params.in_size_; i++) {
        next_state_ += U.host_at(i * params.out_size_ + o) * in.host_at(i);
      }

      if (params.has_bias_) {
        next_state_ += bias.host_at(o);
      }
      next_state.host_at(o) = next_state_;
    }

    // TODO(Randl): temporary
    vec_t t1 = next_state.toVec();
    params.activation_->forward_activation(t1, t1);
    next_state.fromVec(t1);

    // V matrix is out_size_ x out_size_
    for (size_t o = 0; o < params.out_size_; o++) {
      float_t out_ = 0;
      for (size_t o_2 = 0; o_2 < params.out_size_; o_2++) {
        out_ += V.host_at(o_2 * params.out_size_ + o) * next_state.host_at(o_2);
      }

      if (params.has_bias_) {
        out_ += c.host_at(o);
      }
      out.host_at(o) = out_;
    }
  });
}

template <typename S1,
          typename S2,
          typename S3,
          typename S4,
          typename S5,
          typename S6,
          typename S7,
          typename S8,
          typename S9,
          typename S10,
          typename S11,
          typename S12,
          typename S13,
          typename S14,
          typename S15>
inline void recurrent_cell_op_internal(
  const Tensor<float_t, S1> &prev_out,
  const Tensor<float_t, S2> &prev_h,
  const Tensor<float_t, S3> &U,
  const Tensor<float_t, S4> &W,
  const Tensor<float_t, S5> &V,
  Tensor<float_t, S6> &dU,
  Tensor<float_t, S7> &dW,
  Tensor<float_t, S8> &dV,
  Tensor<float_t, S9> &db,
  Tensor<float_t, S10> &dc,
  const Tensor<float_t, S11> &curr_output_delta,
  Tensor<float_t, S12> &curr_state_delta,
  Tensor<float_t, S13> &prev_output_delta,
  Tensor<float_t, S14> &prev_state_delta,
  const Tensor<float_t, S15> &out_h,
  const core::recurrent_cell_params &params,
  const bool layer_parallelize) {
  for (size_t sample = 0; sample < prev_out.size(); sample++) {
    const auto prev_out_ =
      prev_out.subView(TensorSingleIndex(sample), TensorAll());
    const auto prev_h_ = prev_h.subView(TensorSingleIndex(sample), TensorAll());
    auto dU_           = dU.subView(TensorSingleIndex(sample), TensorAll());
    auto dW_           = dW.subView(TensorSingleIndex(sample), TensorAll());
    auto dV_           = dV.subView(TensorSingleIndex(sample), TensorAll());
    auto db_           = db.subView(TensorSingleIndex(sample), TensorAll());
    auto dc_           = dc.subView(TensorSingleIndex(sample), TensorAll());
    const auto curr_output_delta_ =
      curr_output_delta.subView(TensorSingleIndex(sample), TensorAll());
    auto curr_state_delta_ =
      curr_state_delta.subView(TensorSingleIndex(sample), TensorAll());
    auto prev_output_delta_ =
      prev_output_delta.subView(TensorSingleIndex(sample), TensorAll());
    auto prev_state_delta_ =
      prev_state_delta.subView(TensorSingleIndex(sample), TensorAll());
    const auto out_h_ = out_h.subView(TensorSingleIndex(sample), TensorAll());

    // from output to h
    for (size_t o = 0; o < params.out_size_; o++) {
      // propagate delta from output to h.
      curr_state_delta_.host_at(o) +=
        vectorize::dot(curr_output_delta_.host_pbegin(),
                       V.host_pointer(o * params.out_size_), params.out_size_);
    }

    // h'(t)
    // TODO(Randl): tmp
    const vec_t prev_h_v = prev_h_.toVec(), out_h_v = out_h_.toVec();
    vec_t curr_state_delta_v = curr_state_delta_.toVec();
    params.activation_->backward_activation(
      prev_h_v, out_h_v, curr_state_delta_v, curr_state_delta_v);
    curr_state_delta_.fromVec(curr_state_delta_v);

    // \delta h(t) -W-> h(t-1)
    for (size_t o = 0; o < params.out_size_; o++) {
      prev_state_delta_.host_at(o) +=
        vectorize::dot(curr_state_delta_.host_pbegin(),
                       W.host_pointer(o * params.out_size_), params.out_size_);
    }

    // \delta h(t) -U-> \delta x(t)
    for (size_t i = 0; i < params.in_size_; i++) {
      prev_output_delta_.host_at(i) +=
        vectorize::dot(curr_state_delta_.host_pbegin(),
                       U.host_pointer(i * params.out_size_), params.out_size_);
    }

    for_(layer_parallelize, 0, size_t(params.out_size_),
         [&](const blocked_range &r) {
           // accumulate weight-step using delta
           // dW[c * out_size + i] += current_delta[i] * prev_out[o]
           const size_t begin  = r.begin();
           const size_t end    = r.end();
           const size_t stride = end - begin;
           for (size_t o = 0; o < params.out_size_; o++) {
             vectorize::muladd(curr_output_delta_.host_pointer(begin),
                               out_h_.host_at(o), stride,
                               dV_.host_pointer(o * params.out_size_ + begin));
           }

           if (params.has_bias_) {
             // vec_t& dc;
             for (size_t o = begin; o < end; o++) {
               dc_.host_at(o) += curr_output_delta_.host_at(o);
             }
           }

           for (size_t o = 0; o < params.out_size_; o++) {
             vectorize::muladd(curr_state_delta_.host_pointer(begin),
                               prev_h_.host_at(o), stride,
                               dW_.host_pointer(o * params.out_size_ + begin));
           }

           for (size_t i = 0; i < params.in_size_; i++) {
             vectorize::muladd(curr_state_delta_.host_pointer(begin),
                               prev_out_.host_at(i), stride,
                               dU_.host_pointer(i * params.out_size_ + begin));
           }

           if (params.has_bias_) {
             // vec_t& db;
             for (size_t o = begin; o < end; o++) {
               db_.host_at(o) += curr_state_delta_.host_at(o);
             }
           }
         });
  }
}

}  // namespace kernels
}  // namespace tiny_dnn
