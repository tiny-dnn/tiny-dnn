/*
    Copyright (c) 2015, Taiga Nomi
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
#include "util.h"
#include "partial_connected_layer.h"
#include "activation_function.h"

namespace tiny_cnn {
    
template <typename Activation>
class max_pooling_layer : public layer<Activation> {
public:
    typedef layer<Activation> Base;

    max_pooling_layer(layer_size_t in_width, layer_size_t in_height, layer_size_t in_channels, layer_size_t pooling_size)
        : layer<Activation>(
        in_width * in_height * in_channels,
        in_width * in_height * in_channels / sqr(pooling_size),
        0, 0),
        in_(in_width, in_height, in_channels),
        out_(in_width / pooling_size, in_height / pooling_size, in_channels)
    {
        if ((in_width % pooling_size) || (in_height % pooling_size))
            throw nn_error("width/height must be multiples of pooling size");
        init_connection(in_width, in_height, in_channels, pooling_size);
    }

    size_t fan_in_size() const override {
        return out2in_[0].size();
    }

    size_t connection_size() const override {
        return out2in_[0].size() * out2in_.size();
    }

    virtual const vec_t& forward_propagation(const vec_t& in, size_t index) {
        for_(this->parallelize_, 0, this->out_size_, [&](const blocked_range& r) {
            for (int i = r.begin(); i < r.end(); i++) {
                const auto& in_index = out2in_[i];
                float_t max_value = std::numeric_limits<float_t>::lowest();
                
                for (auto j : in_index) {
                    if (in[j] > max_value) {
                        max_value = in[j];
                        out2inmax_[i] = j;
                    }
                }
                this->output_[index][i] = max_value;
            }
        });
        return this->next_ ? this->next_->forward_propagation(this->output_[index], index) : this->output_[index];
    }

    virtual const vec_t& back_propagation(const vec_t& current_delta, size_t index) {
        const vec_t& prev_out = this->prev_->output(index);
        const activation::function& prev_h = this->prev_->activation_function();
        vec_t& prev_delta = this->prev_delta_[index];

        for_(this->parallelize_, 0, this->in_size_, [&](const blocked_range& r) {
            for (int i = r.begin(); i != r.end(); i++) {
                int outi = in2out_[i];
                prev_delta[i] = (out2inmax_[outi] == i) ? current_delta[outi] * prev_h.df(prev_out[i]) : 0.0;
            }
        });
        return this->prev_->back_propagation(this->prev_delta_[index], index);
    }

    const vec_t& back_propagation_2nd(const vec_t& current_delta2) {
        const vec_t& prev_out = this->prev_->output(0);
        const activation::function& prev_h = this->prev_->activation_function();

        for (int i = 0; i < this->in_size_; i++) {
            int outi = in2out_[i];
            this->prev_delta2_[i] = (out2inmax_[outi] == i) ? current_delta2[outi] * sqr(prev_h.df(prev_out[i])) : 0.0;
        }
        return this->prev_->back_propagation_2nd(this->prev_delta2_);
    }
private:
    std::vector<std::vector<int> > out2in_; // mapping out => in (1:N)
    std::vector<int> in2out_; // mapping in => out (N:1)
    std::vector<int> out2inmax_; // mapping out => max_index(in) (1:1)
    index3d<layer_size_t> in_;
    index3d<layer_size_t> out_;

    void connect_kernel(layer_size_t pooling_size, layer_size_t outx, layer_size_t outy, layer_size_t  c)
    {
        for (layer_size_t dy = 0; dy < pooling_size; dy++) {
            for (layer_size_t dx = 0; dx < pooling_size; dx++) {
                layer_size_t in_index = in_.get_index(outx * pooling_size + dx, outy * pooling_size + dy, c);
                layer_size_t out_index = out_.get_index(outx, outy, c);
                in2out_[in_index] = out_index;
                out2in_[out_index].push_back(in_index);
            }
        }
    }

    void init_connection(layer_size_t in_width, layer_size_t in_height, layer_size_t in_channels, layer_size_t pooling_size)
    {
        in2out_.resize(in_.size());
        out2in_.resize(out_.size());
        out2inmax_.resize(out_.size());
        for (layer_size_t c = 0; c < in_.depth_; ++c)
            for (layer_size_t y = 0; y < out_.height_; ++y)
                for (layer_size_t x = 0; x < out_.width_; ++x)
                    connect_kernel(pooling_size, x, y, c);
    }

};

} // namespace tiny_cnn
