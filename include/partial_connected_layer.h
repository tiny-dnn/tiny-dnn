#pragma once
#include "util.h"
#include "layer.h"

namespace tiny_cnn {

template<typename N, typename Activation>
class partial_connected_layer : public layer<N, Activation> {
public:
    typedef std::vector<std::pair<int, int> > io_connections;
    typedef std::vector<std::pair<int, int> > wi_connections;
    typedef std::vector<std::pair<int, int> > wo_connections;
    typedef layer<N, Activation> Base;
    typedef typename Base::Updater Updater;

    partial_connected_layer(int in_dim, int out_dim, int weight_dim, int bias_dim, float_t scale_factor = 1.0)
        : layer<N, Activation> (in_dim, out_dim, weight_dim, bias_dim), 
        weight2io_(weight_dim), out2wi_(out_dim), in2wo_(in_dim), bias2out_(bias_dim), out2bias_(out_dim), scale_factor_(scale_factor) {
        if (in_dim <= 0 || weight_dim <= 0 || weight_dim <= 0 || bias_dim <= 0)
            throw nn_error("invalid layer size");
    }

    int param_size() const { 
        int total_param = 0;
        for (auto w : weight2io_)
            if (w.size() > 0) total_param++;
        for (auto b : bias2out_)
            if (b.size() > 0) total_param++;
        return total_param;
    }

    int connection_size() const {
        int total_size = 0;
        for (auto io : weight2io_)
            total_size += io.size();
        for (auto b : bias2out_)
            total_size += b.size();
        return total_size;
    }

    int fan_in_size() const {
        return out2wi_[0].size();
    }

    void connect_weight(int input_index, int output_index, int weight_index) {
        weight2io_[weight_index].push_back(std::make_pair(input_index, output_index));
        out2wi_[output_index].push_back(std::make_pair(weight_index, input_index));
        in2wo_[input_index].push_back(std::make_pair(weight_index, output_index));
    }

    void connect_bias(int bias_index, int output_index) {
        out2bias_[output_index] = bias_index;
        bias2out_[bias_index].push_back(output_index);
    }

    virtual const vec_t& forward_propagation(const vec_t& in) {

        parallel_for(0, this->out_size_, [&](const blocked_range& r) {
            for (int i = r.begin(); i < r.end(); i++) {
                const wi_connections& connections = out2wi_[i];
                float_t a = 0.0;

                for (auto& connection : connections)// 13.1%
                    a += this->W_[connection.first] * in[connection.second]; // 3.2%

                a *= scale_factor_;
                a += this->b_[out2bias_[i]];
                this->output_[i] = this->a_.f(a); // 9.6%
            }
        });

        return this->next_ ? this->next_->forward_propagation(this->output_) : this->output_; // 15.6%
    }

    virtual const vec_t& back_propagation(const vec_t& current_delta, Updater *l) {
        const vec_t& prev_out = this->prev_->output();
        const activation& prev_h = this->prev_->activation_function();

        parallel_for(0, this->in_size_, [&](const blocked_range& r) {
            for (int i = r.begin(); i != r.end(); i++) {
                const wo_connections& connections = in2wo_[i];
                this->prev_delta_[i] = 0.0;

                for (auto connection : connections) 
                    this->prev_delta_[i] += this->W_[connection.first] * current_delta[connection.second]; // 40.6%

                this->prev_delta_[i] *= scale_factor_ * prev_h.df(prev_out[i]); // 2.1%
            }
        });

        if (l) {
            parallel_for(0, weight2io_.size(), [&](const blocked_range& r) {
                for (int i = r.begin(); i < r.end(); i++) {
                    const io_connections& connections = weight2io_[i];
                    float_t diff = 0.0;

                    for (auto connection : connections) // 11.9%
                        diff += prev_out[connection.first] * current_delta[connection.second];

                    diff *= scale_factor_;
                    l->update(diff, this->Whessian_[i], &this->W_[i]);// 9.8%
                }
            });

            for (size_t i = 0; i < bias2out_.size(); i++) {
                const std::vector<int>& outs = bias2out_[i];
                float_t diff = 0.0;

                for (auto o : outs)
                    diff += current_delta[o];    

                l->update(diff, this->bhessian_[i], &this->b_[i]); 
            }
        }

        return this->prev_->back_propagation(this->prev_delta_, l);
    }

    const vec_t& back_propagation_2nd(const vec_t& current_delta2) {
        const vec_t& prev_out = this->prev_->output();
        const activation& prev_h = this->prev_->activation_function();

        for (size_t i = 0; i < weight2io_.size(); i++) {
            const io_connections& connections = weight2io_[i];
            float_t diff = 0.0;

            for (auto connection : connections)
                diff += prev_out[connection.first] * prev_out[connection.first] * current_delta2[connection.second];

            diff *= scale_factor_ * scale_factor_;
            this->Whessian_[i] += diff;
        }

        for (size_t i = 0; i < bias2out_.size(); i++) {
            const std::vector<int>& outs = bias2out_[i];
            float_t diff = 0.0;

            for (auto o : outs)
                diff += current_delta2[o];    

            this->bhessian_[i] += diff;
        }

        for (int i = 0; i < this->in_size_; i++) {
            const wo_connections& connections = in2wo_[i];
            this->prev_delta2_[i] = 0.0;

            for (auto connection : connections) 
                this->prev_delta2_[i] += this->W_[connection.first] * this->W_[connection.first] * current_delta2[connection.second];

            this->prev_delta2_[i] *= scale_factor_ * scale_factor_ * prev_h.df(prev_out[i]) * prev_h.df(prev_out[i]);
        }
        return this->prev_->back_propagation_2nd(this->prev_delta2_);
    }

    // remove unused weight to improve cache hits
    void remap() {
        std::map<int, int> swaps;
        int n = 0;

        for (size_t i = 0; i < weight2io_.size(); i++)
            swaps[i] = weight2io_[i].empty() ? -1 : n++;

        for (int i = 0; i < this->out_size_; i++) {
            wi_connections& wi = out2wi_[i];
            for (size_t j = 0; j < wi.size(); j++)
                wi[j].first = swaps[wi[j].first];
        }

        for (int i = 0; i < this->in_size_; i++) {
            wo_connections& wo = in2wo_[i];
            for (size_t j = 0; j < wo.size(); j++)
                wo[j].first = swaps[wo[j].first];
        }

        std::vector<io_connections> weight2io_new(n);
        for (size_t i = 0; i < weight2io_.size(); i++)
            if(swaps[i] >= 0) weight2io_new[swaps[i]] = weight2io_[i];

        weight2io_ = weight2io_new;
    }

protected:
    std::vector<io_connections> weight2io_; // weight_id -> [(in_id, out_id)]
    std::vector<wi_connections> out2wi_; // out_id -> [(weight_id, in_id)]
    std::vector<wo_connections> in2wo_; // in_id -> [(weight_id, out_id)]
    std::vector<std::vector<int> > bias2out_;
    std::vector<int> out2bias_;
    float_t scale_factor_;
};

} 