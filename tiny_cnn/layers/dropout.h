/*
    Copyright (c) 2013, Taiga Nomi
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
#include "tiny_cnn/util/product.h"
#include "tiny_cnn/util/util.h"

namespace tiny_cnn {

    class filter_none {
    public:
        explicit filter_none(int out_dim) {
            CNN_UNREFERENCED_PARAMETER(out_dim);
        }

        const vec_t& filter_fprop(const vec_t& out, int index) {
            CNN_UNREFERENCED_PARAMETER(index);
            return out;
        }

        const vec_t& filter_bprop(const vec_t& delta, int index) {
            CNN_UNREFERENCED_PARAMETER(index);
            return delta;
        }
    };

    class dropout {
    public:
        enum context {
            train_phase,
            test_phase
        };

        enum mode {
            per_data,
            per_batch
        };

        explicit dropout(int out_dim)
            : out_dim_(out_dim), mask_(out_dim), ctx_(train_phase), mode_(per_data), dropout_rate_(0.5) {
            for (int i = 0; i < CNN_TASK_SIZE; i++) {
                masked_out_[i].resize(out_dim);
                masked_delta_[i].resize(out_dim);
            }
            shuffle();
        }

        void set_dropout_rate(double rate) {
            if (rate < 0.0 || rate >= 1.0)
                throw nn_error("0.0 <= dropout-rate < 1.0");
            dropout_rate_ = rate;
        }

        void set_mode(mode mode) {
            mode_ = mode;
        }

        void set_context(context ctx) {
            ctx_ = ctx;
        }

        // mask output vector
        const vec_t& filter_fprop(const vec_t& out, int index) {
            if (ctx_ == train_phase) {
                for (int i = 0; i < out_dim_; i++)
                    masked_out_[index][i] = out[i] * mask_[i];
            }
            else if (ctx_ == test_phase) {
                for (int i = 0; i < out_dim_; i++)
                    masked_out_[index][i] = out[i] * (1.0 - dropout_rate_);
            }
            else {
                throw nn_error("invalid context");
            }
            return masked_out_[index];
        }

        // mask delta
        const vec_t& filter_bprop(const vec_t& delta, int index) {
            for (int i = 0; i < out_dim_; i++)
                masked_delta_[index][i] = delta[i] * mask_[i];

            if (mode_ == per_data) shuffle();

            return masked_delta_[index];
        }

        void shuffle() {
            for (auto& m : mask_)
                m = bernoulli(1.0 - dropout_rate_);
        }

        void end_batch() {
            if (mode_ == per_batch) shuffle();
        }

    private:
        int out_dim_;
        std::vector<uint8_t> mask_;
        vec_t masked_out_[CNN_TASK_SIZE];
        vec_t masked_delta_[CNN_TASK_SIZE];
        context ctx_;
        mode mode_;
        double dropout_rate_;
    };

} // namespace tiny_cnn
