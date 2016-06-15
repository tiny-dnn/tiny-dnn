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

#include "tiny_cnn/core/math_backend.h"

namespace tiny_cnn {
namespace core {

class tiny_backend : public math_backend {
 public:
  // context holds solution-dependent parameters
  // context should be able to hold any types of structures (like boost::any)
  tiny_backend(conv_params* params,
               std::function<void(const vec_t&, int)> f1,
               std::function<void(const vec_t&, vec_t&)> f2,
               std::function<void(const vec_t&, const vec_t&, vec_t&)> f3,
               std::vector<conv_layer_worker_specific_storage>* ptr)
      : params_(params), conv_layer_worker_storage_(ptr),
        copy_and_pad_input(f1), copy_and_unpad_delta(f2),
        backward_activation(f3) {}

  // core math functions

  void conv2d(cnn_size_t                 index,
              const std::vector<vec_t*>& in_data,
              std::vector<vec_t*>&       out_data) {
      copy_and_pad_input(*in_data[0], static_cast<int>(index));
      const vec_t& W   = *in_data[1];
      vec_t&       a   = *out_data[1];
      const vec_t &in  = *((*conv_layer_worker_storage_)[index].prev_out_padded_); // input // NOLINT

      std::fill(a.begin(), a.end(), float_t(0));

      for_i(layer_->get_parallelize(), params_->out.depth_, [&](int o) {
          for (cnn_size_t inc = 0; inc < params_->in.depth_; inc++) {
              if (!params_->tbl.is_connected(o, inc)) continue;

              cnn_size_t idx = 0;
              idx = params_->in.depth_ * o + inc;
              idx = params_->weight.get_index(0, 0, idx);
              const float_t *pw = &W[idx];

              idx = params_->in_padded.get_index(0, 0, inc);
              const float_t *pi = &in[idx];

              idx = params_->out.get_index(0, 0, o);
              float_t *pa = &a[idx];

              for (cnn_size_t y = 0; y < params_->out.height_; y++) {
                  for (cnn_size_t x = 0; x < params_->out.width_; x++) {
                      const float_t * ppw = pw;
                      const float_t * ppi = pi + params_->in_padded.width_ *
                                            (y * params_->h_stride) +
                                             x * params_->w_stride;
                      float_t sum = float_t(0);

                      // should be optimized for small kernel(3x3,5x5)
                      for (cnn_size_t wy = 0; wy < params_->weight.height_; wy++) {    // NOLINT
                          for (cnn_size_t wx = 0; wx < params_->weight.width_; wx++) { // NOLINT
                              idx = wy * params_->in_padded.width_ + wx;
                              sum += *ppw++ * ppi[idx];
                          }
                      }
                      pa[y * params_->out.width_ + x] += sum;
                  }
              }
          }

          if (params_->has_bias) {
              const vec_t& bias = *in_data[2];
              float_t * pa  = &a[params_->out.get_index(0, 0, o)];
              float_t * paa = pa + params_->out.width_ * params_->out.height_;
              std::for_each(pa, paa, [&](float_t& f) { f += bias[o]; });
          }
      });

      /*for_i(this->parallelize_, params.out.size(), [&](int i) {
          out[i] = this->h_.f(a, i);
      });*/
  }

  void conv2d_back(cnn_size_t                 index,
                   const std::vector<vec_t*>& in_data,
                   const std::vector<vec_t*>& out_data,
                   std::vector<vec_t*>&       out_grad,
                   std::vector<vec_t*>&       in_grad) {
      conv_layer_worker_specific_storage& cws =
         (*conv_layer_worker_storage_)[index];

      const vec_t& prev_out = *(cws.prev_out_padded_);
      const vec_t& W  = *in_data[1];
      vec_t&       dW = *in_grad[1];
      vec_t&       curr_delta = *out_grad[1];
      vec_t*       prev_delta = (params_->pad_type == padding::same) ?
                                 &cws.prev_delta_padded_ : in_grad[0];

      assert(W.size() == params_->weight.size());
      assert(dW.size() == params_->weight.size());
      assert(curr_delta.size() ==  layer_->out_shape()[0].size());

      // this->backward_activation(*out_grad[0], *out_data[0], curr_delta);
      backward_activation(*out_grad[0], *out_data[0], curr_delta);

      std::fill(prev_delta->begin(), prev_delta->end(), float_t(0));

      // propagate delta to previous layer
      for_i(params_->in.depth_, [&](int inc) {
         for (cnn_size_t outc = 0; outc < params_->out.depth_; outc++) {
             if (!params_->tbl.is_connected(outc, inc)) continue;

             cnn_size_t idx = 0;
             idx = params_->in.depth_ * outc + inc;
             idx = params_->weight.get_index(0, 0, idx);
             const float_t *pw = &W[idx];

             idx = params_->out.get_index(0, 0, outc);
             const float_t *pdelta_src = &curr_delta[idx];

             idx = params_->in_padded.get_index(0, 0, inc);
             float_t *pdelta_dst = &(*prev_delta)[idx];

             for (cnn_size_t y = 0; y < params_->out.height_; y++) {
                 for (cnn_size_t x = 0; x < params_->out.width_; x++) {
                     const float_t * ppw = pw;

                     idx = y * params_->out.width_ + x;
                     const float_t ppdelta_src = pdelta_src[idx];

                     float_t * ppdelta_dst = pdelta_dst +
                         y * params_->h_stride * params_->in_padded.width_ +
                         x * params_->w_stride;

                     for (cnn_size_t wy = 0; wy < params_->weight.height_; wy++) {    // NOLINT
                         for (cnn_size_t wx = 0; wx < params_->weight.width_; wx++) { // NOLINT
                             idx = wy * params_->in_padded.width_ + wx;
                             ppdelta_dst[idx] += *ppw++ * ppdelta_src;
                         }
                     }
                 }
             }
         }
      });

      // accumulate dw
      for_i(params_->in.depth_, [&](int inc) {
         for (cnn_size_t outc = 0; outc < params_->out.depth_; outc++) {
             if (!params_->tbl.is_connected(outc, inc)) continue;

             for (cnn_size_t wy = 0; wy < params_->weight.height_; wy++) {
                 for (cnn_size_t wx = 0; wx < params_->weight.width_; wx++) {
                     float_t dst = float_t(0);

                     cnn_size_t idx = 0;
                     idx = params_->in_padded.get_index(wx, wy, inc);
                     const float_t * prevo = &prev_out[idx];

                     idx = params_->out.get_index(0, 0, outc);
                     const float_t * delta = &curr_delta[idx];

                     for (cnn_size_t y = 0; y < params_->out.height_; y++) {
                         dst += vectorize::dot(
                             prevo + y * params_->in_padded.width_,
                             delta + y * params_->out.width_,
                             params_->out.width_);
                     }

                     idx = params_->in.depth_ * outc + inc;
                     dW[params_->weight.get_index(wx, wy, idx)] += dst;
                 }
             }
         }
      });

      // accumulate db
      if (params_->has_bias) {
         vec_t& db = *in_grad[2];

         for (cnn_size_t outc = 0; outc < params_->out.depth_; outc++) {
             cnn_size_t idx = params_->out.get_index(0, 0, outc);
             const float_t * delta = &curr_delta[idx];
             const float_t * deltaa = delta + params_->out.width_ *
                                              params_->out.height_;
             db[outc] += std::accumulate(delta, deltaa, float_t(0));
         }
      }

      if (params_->pad_type == padding::same) {
         copy_and_unpad_delta(cws.prev_delta_padded_, *in_grad[0]);
      }
  }

    void matmul() {
        throw nn_error("not implemented yet.");
    }

    void maxpool() {
        throw nn_error("not implemented yet.");
    }

 private:
    /* Pointer to the convolution parameters */
    conv_params* params_;

    /* Pointer to the convolution workers */
    std::vector<conv_layer_worker_specific_storage>* conv_layer_worker_storage_;

    /* Pointers to parent class functions */
    std::function<void(const vec_t&, int)> copy_and_pad_input;
    std::function<void(const vec_t&, vec_t&)> copy_and_unpad_delta;
    std::function<void(const vec_t&, const vec_t&, vec_t&)> backward_activation;
};

}  // namespace core
}  // namespace tiny_cnn
