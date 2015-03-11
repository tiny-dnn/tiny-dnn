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
#include "util.h"
#include "layer.h"

namespace tiny_cnn {

template<typename N, typename Activation>
class partial_connected_layer : public layer<N, Activation> {
public:
    using io_connections = std::vector<std::pair<layer_size_t, layer_size_t>>;
    using wi_connections = std::vector<std::pair<layer_size_t, layer_size_t>>;
    using wo_connections = std::vector<std::pair<layer_size_t, layer_size_t>>;
    using Base = layer<N, Activation>;
    using Optimizer = typename layer<N, Activation>::Optimizer;

    partial_connected_layer(layer_size_t in_dim, layer_size_t out_dim, size_t weight_dim, size_t bias_dim, float_t scale_factor = 1.0)
        : layer<N, Activation> (in_dim, out_dim, weight_dim, bias_dim), 
        weight2io_(weight_dim), out2wi_(out_dim), in2wo_(in_dim), bias2out_(bias_dim), out2bias_(out_dim), scale_factor_(scale_factor) {}

    size_t param_size() const override {
        return
            std::accumulate(cbegin(weight2io_), cend(weight2io_), size_t{}, [](auto value, const auto &io) { return io.size() > 0 ? value + 1 : value; }) +
            std::accumulate(cbegin(bias2out_), cend(bias2out_), size_t{}, [](auto value, const auto &b) { return b.size() > 0 ? value + 1 : value; });
    }

    size_t connection_size() const override {
        return
            std::accumulate(cbegin(weight2io_), cend(weight2io_), size_t{}, [](auto value, const auto &io) { return value + io.size(); }) +
            std::accumulate(cbegin(bias2out_), cend(bias2out_), size_t{}, [](auto value, const auto &b) { return value + b.size(); });
    }

    size_t fan_in_size() const override {
        return out2wi_[0].size();
    }

    void connect_weight(layer_size_t input_index, layer_size_t output_index, layer_size_t weight_index) {
        weight2io_[weight_index].emplace_back(input_index, output_index);
        out2wi_[output_index].emplace_back(weight_index, input_index);
        in2wo_[input_index].emplace_back(weight_index, output_index);
    }

    void connect_bias(layer_size_t bias_index, layer_size_t output_index) {
        out2bias_[output_index] = bias_index;
        bias2out_[bias_index].push_back(output_index);
    }

    vec_t forward_propagation(const vec_t& in, size_t index) override {

        for_(this->parallelize_, 0, this->out_size_, [&](const blocked_range& r) {
            for (size_t i = r.begin(); i < r.end(); i++) {
                this->output_[index][i] = this->a_.f(this->b_[out2bias_[i]] + scale_factor_ * std::accumulate(cbegin(out2wi_[i]), cend(out2wi_[i]), float_t{}, [&](auto value, const auto &connection) { return value + this->W_[connection.first] * in[connection.second]; }));
            }
        });

        return this->next_ ? this->next_->forward_propagation(this->output_[index], index) : this->output_[index]; // 15.6%
    }

    vec_t back_propagation(vec_t&& current_delta, size_t index) override {
        const vec_t& prev_out = this->prev_->output(index);
        const activation::function& prev_h = this->prev_->activation_function();
        vec_t prev_delta(this->in_size_);

        for_(this->parallelize_, 0, this->in_size_, [&](const blocked_range& r) {
            for (size_t i = r.begin(); i < r.end(); i++) {
                prev_delta[i] = scale_factor_ * prev_h.df(prev_out[i]) * std::accumulate(cbegin(in2wo_[i]), cend(in2wo_[i]), float_t{}, [&](auto value, auto connection) { return value + this->W_[connection.first] * current_delta[connection.second]; });
            }
        });

        for_(this->parallelize_, 0, weight2io_.size(), [&](const blocked_range& r) {
            for (size_t i = r.begin(); i < r.end(); i++) {
                this->dW_[index][i] += scale_factor_ * std::accumulate(cbegin(weight2io_[i]), cend(weight2io_[i]), float_t{}, [&](auto value, auto connection) { return value + prev_out[connection.first] * current_delta[connection.second]; });
            }
        });

        for (size_t i = 0; i < bias2out_.size(); i++) {
            this->db_[index][i] += std::accumulate(cbegin(bias2out_[i]), cend(bias2out_[i]), float_t{}, [&](auto value, auto o) { return value + current_delta[o]; });
        } 

        return this->prev_->back_propagation(move(prev_delta), index);
    }

    vec_t back_propagation_2nd(vec_t&& current_delta2) override {
        const vec_t& prev_out = this->prev_->output(0);
        const activation::function& prev_h = this->prev_->activation_function();

        for (size_t i = 0; i < weight2io_.size(); i++) {
            this->Whessian_[i] += sqr(scale_factor_)*std::accumulate(cbegin(weight2io_[i]), cend(weight2io_[i]), float_t{}, [&](auto value, auto weightio) { return value + sqr(prev_out[weightio.first]) * current_delta2[weightio.second]; });
        }

        for (size_t i = 0; i < bias2out_.size(); i++) {
            this->bhessian_[i] += std::accumulate(cbegin(bias2out_[i]), cend(bias2out_[i]), float_t{}, [&](auto value, auto index) { return value + current_delta2[index]; });
        }

        vec_t prev_delta2;
        prev_delta2.reserve(this->in_size_);
        for (size_t i = 0; i < this->in_size_; i++) {
            
            prev_delta2.push_back(std::accumulate(cbegin(in2wo_[i]), cend(in2wo_[i]), float_t{}, [&](auto value, auto connection) { return value + sqr(this->W_[connection.first]) * current_delta2[connection.second]; }) * sqr(scale_factor_ * prev_h.df(prev_out[i])));
        }
        return this->prev_->back_propagation_2nd(move(prev_delta2));
    }

    // remove unused weight to improve cache hits
    void remap() {
        std::map<int, int> swaps;
        size_t n = 0;

        for (size_t i = 0; i < weight2io_.size(); i++)
            swaps[i] = weight2io_[i].empty() ? -1 : n++;

        for (size_t i = 0; i < this->out_size_; i++) {
            wi_connections& wi = out2wi_[i];
            for (size_t j = 0; j < wi.size(); j++)
                wi[j].first = static_cast<unsigned short>(swaps[wi[j].first]);
        }

        for (size_t i = 0; i < this->in_size_; i++) {
            wo_connections& wo = in2wo_[i];
            for (size_t j = 0; j < wo.size(); j++)
                wo[j].first = static_cast<unsigned short>(swaps[wo[j].first]);
        }

        std::vector<io_connections> weight2io_new(n);
        for (size_t i = 0; i < weight2io_.size(); i++)
            if(swaps[i] >= 0) weight2io_new[swaps[i]] = weight2io_[i];

        weight2io_.swap(weight2io_new);
    }

protected:
    std::vector<io_connections> weight2io_; // weight_id -> [(in_id, out_id)]
    std::vector<wi_connections> out2wi_; // out_id -> [(weight_id, in_id)]
    std::vector<wo_connections> in2wo_; // in_id -> [(weight_id, out_id)]
    std::vector<std::vector<layer_size_t> > bias2out_;
    std::vector<size_t> out2bias_;
    float_t scale_factor_;
};

} // namespace tiny_cnn
