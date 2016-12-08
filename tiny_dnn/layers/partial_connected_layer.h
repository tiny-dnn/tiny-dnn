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
#include "tiny_dnn/util/util.h"
#include "tiny_dnn/layers/layer.h"

namespace tiny_dnn {

template<typename Activation>
class partial_connected_layer : public feedforward_layer<Activation> {
public:
    CNN_USE_LAYER_MEMBERS;

    typedef std::vector<std::pair<serial_size_t, serial_size_t> > io_connections;
    typedef std::vector<std::pair<serial_size_t, serial_size_t> > wi_connections;
    typedef std::vector<std::pair<serial_size_t, serial_size_t> > wo_connections;
    typedef feedforward_layer<Activation> Base;

    partial_connected_layer(serial_size_t in_dim,
                            serial_size_t out_dim,
                            size_t     weight_dim,
                            size_t     bias_dim,
                            float_t    scale_factor = float_t(1))
        : Base(std_input_order(bias_dim > 0)),
          weight2io_(weight_dim),
          out2wi_(out_dim),
          in2wo_(in_dim),
          bias2out_(bias_dim),
          out2bias_(out_dim),
          scale_factor_(scale_factor){}

    size_t param_size() const {
        size_t total_param = 0;
        for (auto w : weight2io_)
            if (w.size() > 0) total_param++;
        for (auto b : bias2out_)
            if (b.size() > 0) total_param++;
        return total_param;
    }

    serial_size_t fan_in_size() const override {
        return max_size(out2wi_);
    }

    serial_size_t fan_out_size() const override {
        return max_size(in2wo_);
    }

    void connect_weight(serial_size_t input_index, serial_size_t output_index, serial_size_t weight_index) {
        weight2io_[weight_index].emplace_back(input_index, output_index);
        out2wi_[output_index].emplace_back(weight_index, input_index);
        in2wo_[input_index].emplace_back(weight_index, output_index);
    }

    void connect_bias(serial_size_t bias_index, serial_size_t output_index) {
        out2bias_[output_index] = bias_index;
        bias2out_[bias_index].push_back(output_index);
    }

    void forward_propagation(const std::vector<tensor_t*>& in_data,
                             std::vector<tensor_t*>& out_data) override {
        const tensor_t& in  = *in_data[0];
        const vec_t&    W   = (*in_data[1])[0];
        const vec_t&    b   = (*in_data[2])[0];
        tensor_t&       a   = *out_data[1];

        // @todo revise the parallelism strategy
        for (serial_size_t sample = 0, sample_count = static_cast<serial_size_t>(in.size()); sample < sample_count; ++sample) {
            vec_t& a_sample = a[sample];

            for_i(parallelize_, out2wi_.size(), [&](int i) {
                const wi_connections& connections = out2wi_[i];

                float_t& a_element = a_sample[i];

                a_element = float_t(0);

                for (auto connection : connections)// 13.1%
                    a_element += W[connection.first] * in[sample][connection.second]; // 3.2%

                a_element *= scale_factor_;
                a_element += b[out2bias_[i]];
            });
        }

        this->forward_activation(*out_data[0], *out_data[1]);
    }

    void back_propagation(const std::vector<tensor_t*>& in_data,
                          const std::vector<tensor_t*>& out_data,
                          std::vector<tensor_t*>&       out_grad,
                          std::vector<tensor_t*>&       in_grad) override {
        const tensor_t& prev_out    = *in_data[0];
        const vec_t&    W           = (*in_data[1])[0];
        vec_t&          dW          = (*in_grad[1])[0];
        vec_t&          db          = (*in_grad[2])[0];
        tensor_t&       prev_delta  = *in_grad[0];
        tensor_t&       curr_delta  = *out_grad[0];

        this->backward_activation(*out_grad[0], *out_data[0], curr_delta);

        // @todo revise the parallelism strategy
        for (serial_size_t sample = 0, sample_count = static_cast<serial_size_t>(prev_out.size()); sample < sample_count; ++sample) {
            for_(parallelize_, 0, in2wo_.size(), [&](const blocked_range& r) {
                for (int i = r.begin(); i != r.end(); i++) {
                    const wo_connections& connections = in2wo_[i];
                    float_t delta = float_t(0);

                    for (auto connection : connections)
                        delta += W[connection.first] * curr_delta[sample][connection.second]; // 40.6%

                    prev_delta[sample][i] = delta * scale_factor_; // 2.1%
                }
            });

            for_(parallelize_, 0, weight2io_.size(), [&](const blocked_range& r) {
                for (int i = r.begin(); i < r.end(); i++) {
                    const io_connections& connections = weight2io_[i];
                    float_t diff = float_t(0);

                    for (auto connection : connections) // 11.9%
                        diff += prev_out[sample][connection.first] * curr_delta[sample][connection.second];

                    dW[i] += diff * scale_factor_;
                }
            });

            for (size_t i = 0; i < bias2out_.size(); i++) {
                const std::vector<serial_size_t>& outs = bias2out_[i];
                float_t diff = float_t(0);

                for (auto o : outs)
                    diff += curr_delta[sample][o];

                db[i] += diff;
            }
        }
    }

protected:
    std::vector<io_connections> weight2io_; // weight_id -> [(in_id, out_id)]
    std::vector<wi_connections> out2wi_; // out_id -> [(weight_id, in_id)]
    std::vector<wo_connections> in2wo_; // in_id -> [(weight_id, out_id)]
    std::vector<std::vector<serial_size_t> > bias2out_;
    std::vector<size_t> out2bias_;
    float_t scale_factor_;
};

} // namespace tiny_dnn
