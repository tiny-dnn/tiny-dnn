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
#include "tiny_cnn/util/util.h"
#include "tiny_cnn/layers/layer.h"
#include <algorithm>

namespace tiny_cnn {

// normal 
class dropout_layer : public layer<activation::identity> {
public:
    typedef activation::identity Activation;
    CNN_USE_LAYER_MEMBERS;

    dropout_layer(cnn_size_t in_dim, float_t dropout_rate, net_phase phase = net_phase::train)
        : layer<activation::identity>(in_dim, in_dim, 0, 0),
          phase_(phase),
          dropout_rate_(dropout_rate),
          scale_(float_t(1) / (float_t(1) - dropout_rate_))
    {
        mask_ = new bool[in_size_ * CNN_TASK_SIZE];
        std::fill(mask_, mask_ + (in_size_ * CNN_TASK_SIZE), false);
    }

    dropout_layer(const dropout_layer& obj)
        : layer<activation::identity>(obj.in_size_, obj.in_size_, 0, 0),
          phase_(obj.phase_),
          dropout_rate_(obj.dropout_rate_),
          scale_(float_t(1) / (float_t(1) - dropout_rate_))
    {
        mask_ = new bool[in_size_ * CNN_TASK_SIZE];
        std::copy(obj.mask_, (obj.mask_ + (in_size_ * CNN_TASK_SIZE)), mask_);
    }

    dropout_layer(dropout_layer&& obj)
        : layer<activation::identity>(obj.in_size_, obj.in_size_, 0, 0),
          phase_(obj.phase_),
          dropout_rate_(obj.dropout_rate_),
          scale_(float_t(1) / (float_t(1) - dropout_rate_)),
          mask_(obj.mask_)
    {
        obj.mask_ = nullptr;
    }

    virtual ~dropout_layer()
    {
        delete[] mask_;
    }

    dropout_layer& operator=(const dropout_layer& obj)
    {
        delete[] mask_;

        layer::operator=(obj);
        phase_ = obj.phase_;
        dropout_rate_ = obj.dropout_rate_;
        scale_ = obj.scale_;
        mask_ = new bool[in_size_ * CNN_TASK_SIZE];
        std::copy(obj.mask_, (obj.mask_ + (obj.in_size_ * CNN_TASK_SIZE)), mask_);
        return *this;
    }

    dropout_layer& operator=(dropout_layer&& obj)
    {
        layer::operator=(obj);
        phase_ = obj.phase_;
        dropout_rate_ = obj.dropout_rate_;
        scale_ = obj.scale_;
        std::swap(mask_, obj.mask_);
        return *this;
    }

    void set_dropout_rate(float_t rate)
    {
        dropout_rate_ = rate;
        scale_ = float_t(1) / (float_t(1) - dropout_rate_);
    }

    ///< number of incoming connections for each output unit
    size_t fan_in_size() const override
    {
        return 1;
    }

    ///< number of outgoing connections for each input unit
    size_t fan_out_size() const override
    {
        return 1;
    }

    ///< number of connections
    size_t connection_size() const override
    {
        return in_size();
    }

    const vec_t& back_propagation_2nd(const vec_t& in_raw) override 
    {
        prev_delta2_ = in_raw;
        return prev_->back_propagation_2nd(prev_delta2_);
    }

    const vec_t& back_propagation(const vec_t& current_delta, size_t worker_index) override 
    {
        vec_t& prev_delta = prev_delta_[worker_index];
        bool* mask = &mask_[worker_index * CNN_TASK_SIZE];

        for (size_t i = 0; i < current_delta.size(); i++) {
            prev_delta[i] = mask[i] * current_delta[i];
        }
        return prev_->back_propagation(prev_delta, worker_index);
    }

    const vec_t& forward_propagation(const vec_t& in, size_t worker_index) override 
    {
        vec_t& out = output_[worker_index];
        vec_t& a = a_[worker_index];
        bool* mask = &mask_[worker_index * CNN_TASK_SIZE];

        if (phase_ == net_phase::train) {
            for (size_t i = 0; i < in.size(); i++)
                mask[i] = bernoulli(dropout_rate_);

            for (size_t i = 0; i < in.size(); i++)
                a[i] = out[i] = mask[i] * scale_ * in[i];
        }
        else {
            for (size_t i = 0; i < in.size(); i++)
                a[i] = out[i] = in[i];
        }
        return next_ ? next_->forward_propagation(out, worker_index) : out;
    }

    /**
     * set dropout-context (training-phase or test-phase)
     **/
    void set_context(net_phase ctx) override
    {
        phase_ = ctx;
    }

    std::string layer_type() const override { return "dropout"; }

    const bool* get_mask() const { return mask_; }

private:
    net_phase phase_;
    float_t dropout_rate_;
    float_t scale_;
    bool *mask_;
};

} // namespace tiny_cnn
