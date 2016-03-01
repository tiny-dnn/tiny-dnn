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

namespace tiny_cnn {

template<typename Activation>
class partial_connected_layer : public layer<Activation> {
public:
    CNN_USE_LAYER_MEMBERS;

    typedef std::vector<std::pair<cnn_size_t, cnn_size_t> > io_connections;
    typedef std::vector<std::pair<cnn_size_t, cnn_size_t> > wi_connections;
    typedef std::vector<std::pair<cnn_size_t, cnn_size_t> > wo_connections;
    typedef layer<Activation> Base;

    partial_connected_layer(cnn_size_t in_dim, cnn_size_t out_dim, size_t weight_dim, size_t bias_dim, float_t scale_factor = float_t(1))
        : Base(in_dim, out_dim, weight_dim, bias_dim), 
          weight2io_(weight_dim), out2wi_(out_dim), in2wo_(in_dim), bias2out_(bias_dim), out2bias_(out_dim),
          scale_factor_(scale_factor) {}

    size_t param_size() const override {
        size_t total_param = 0;
        for (auto w : weight2io_)
            if (w.size() > 0) total_param++;
        for (auto b : bias2out_)
            if (b.size() > 0) total_param++;
        return total_param;
    }

    size_t connection_size() const override {
        size_t total_size = 0;
        for (auto io : weight2io_)
            total_size += io.size();
        for (auto b : bias2out_)
            total_size += b.size();
        return total_size;
    }

    size_t fan_in_size() const override {
        return max_size(out2wi_);
    }

    size_t fan_out_size() const override {
        return max_size(in2wo_);
    }

    void connect_weight(cnn_size_t input_index, cnn_size_t output_index, cnn_size_t weight_index) {
        weight2io_[weight_index].emplace_back(input_index, output_index);
        out2wi_[output_index].emplace_back(weight_index, input_index);
        in2wo_[input_index].emplace_back(weight_index, output_index);
    }

    void connect_bias(cnn_size_t bias_index, cnn_size_t output_index) {
        out2bias_[output_index] = bias_index;
        bias2out_[bias_index].push_back(output_index);
    }

    const vec_t& forward_propagation(const vec_t& in, size_t index) override {
        vec_t& a = a_[index];
     
        for_i(parallelize_, out_size_, [&](int i) {
            const wi_connections& connections = out2wi_[i];

            a[i] = float_t(0);

            for (auto connection : connections)// 13.1%
                a[i] += W_[connection.first] * in[connection.second]; // 3.2%

            a[i] *= scale_factor_;
            a[i] += b_[out2bias_[i]];
        });

        for_i(parallelize_, out_size_, [&](int i) {
            output_[index][i] = h_.f(a, i);
        });
        CNN_LOG_VECTOR(in, "[pc]in");
        CNN_LOG_VECTOR(W_, "[pc]w");
        CNN_LOG_VECTOR(a, "[pc]a");
        CNN_LOG_VECTOR(output_[index], "[pc]forward");

        return next_ ? next_->forward_propagation(output_[index], index) : output_[index]; // 15.6%
    }

    virtual const vec_t& back_propagation(const vec_t& current_delta, size_t index) override {
        const vec_t& prev_out = prev_->output(index);
        const activation::function& prev_h = prev_->activation_function();
        vec_t& prev_delta = prev_delta_[index];

        for_(parallelize_, 0, size_t(in_size_), [&](const blocked_range& r) {
            for (int i = r.begin(); i != r.end(); i++) {
                const wo_connections& connections = in2wo_[i];
                float_t delta = float_t(0);

                for (auto connection : connections) 
                    delta += W_[connection.first] * current_delta[connection.second]; // 40.6%

                prev_delta[i] = delta * scale_factor_ * prev_h.df(prev_out[i]); // 2.1%
            }
        });

        for_(parallelize_, 0, weight2io_.size(), [&](const blocked_range& r) {
            for (int i = r.begin(); i < r.end(); i++) {
                const io_connections& connections = weight2io_[i];
                float_t diff = float_t(0);

                for (auto connection : connections) // 11.9%
                    diff += prev_out[connection.first] * current_delta[connection.second];

                dW_[index][i] += diff * scale_factor_;
            }
        });

        for (size_t i = 0; i < bias2out_.size(); i++) {
            const std::vector<cnn_size_t>& outs = bias2out_[i];
            float_t diff = float_t(0);

            for (auto o : outs)
                diff += current_delta[o];    

            db_[index][i] += diff;
        } 

        CNN_LOG_VECTOR(current_delta, "[pc]curr_delta");
        CNN_LOG_VECTOR(prev_delta_[index], "[pc]prev_delta");
        CNN_LOG_VECTOR(dW_[index], "[pc]dW");
        CNN_LOG_VECTOR(db_[index], "[pc]db");

        return prev_->back_propagation(prev_delta_[index], index);
    }

    const vec_t& back_propagation_2nd(const vec_t& current_delta2) override {
        const vec_t& prev_out = prev_->output(0);
        const activation::function& prev_h = prev_->activation_function();

        for (size_t i = 0; i < weight2io_.size(); i++) {
            const io_connections& connections = weight2io_[i];
            float_t diff = float_t(0);

            for (auto connection : connections)
                diff += sqr(prev_out[connection.first]) * current_delta2[connection.second];

            diff *= sqr(scale_factor_);
            Whessian_[i] += diff;
        }

        for (size_t i = 0; i < bias2out_.size(); i++) {
            const std::vector<cnn_size_t>& outs = bias2out_[i];
            float_t diff = float_t(0);

            for (auto o : outs)
                diff += current_delta2[o];    

            bhessian_[i] += diff;
        }

        for (cnn_size_t i = 0; i < in_size_; i++) {
            const wo_connections& connections = in2wo_[i];
            prev_delta2_[i] = float_t(0);

            for (auto connection : connections) 
                prev_delta2_[i] += sqr(W_[connection.first]) * current_delta2[connection.second];

            prev_delta2_[i] *= sqr(scale_factor_ * prev_h.df(prev_out[i]));
        }

        CNN_LOG_VECTOR(current_delta2, "[pc]curr-delta2");
        CNN_LOG_VECTOR(prev_delta2_, "[pc]prev-delta2");
        CNN_LOG_VECTOR(Whessian_, "[pc]whessian");

        return prev_->back_propagation_2nd(prev_delta2_);
    }

    // remove unused weight to improve cache hits
    void remap() {
        std::map<int, int> swaps;
        size_t n = 0;

        for (size_t i = 0; i < weight2io_.size(); i++)
            swaps[i] = weight2io_[i].empty() ? -1 : n++;

        for (size_t i = 0; i < out_size_; i++) {
            wi_connections& wi = out2wi_[i];
            for (size_t j = 0; j < wi.size(); j++)
                wi[j].first = static_cast<cnn_size_t>(swaps[wi[j].first]);
        }

        for (size_t i = 0; i < in_size_; i++) {
            wo_connections& wo = in2wo_[i];
            for (size_t j = 0; j < wo.size(); j++)
                wo[j].first = static_cast<cnn_size_t>(swaps[wo[j].first]);
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
    std::vector<std::vector<cnn_size_t> > bias2out_;
    std::vector<size_t> out2bias_;
    float_t scale_factor_;
};

} // namespace tiny_cnn
