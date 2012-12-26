#pragma once
#include "util.h"
#include "layer.h"

namespace nn {

template<typename Activation>
class partial_connected_layer : public layer<Activation> {
public:
    typedef std::vector<std::pair<int, int> > io_connections;
    typedef std::vector<std::pair<int, int> > wi_connections;
    typedef std::vector<std::pair<int, int> > wo_connections;

    partial_connected_layer(int in_dim, int out_dim, int weight_dim, int bias_dim, float_t scale_factor = 1.0)
        : layer<Activation> (in_dim, out_dim, weight_dim, bias_dim), 
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
        for (int i = 0; i < out_size_; i++) {
            const wi_connections& connections = out2wi_[i];
            float_t a = 0.0;

            for (auto connection : connections)
                a += W_[connection.first] * in[connection.second];

            a *= scale_factor_;
            a += b_[out2bias_[i]];
            output_[i] = a_.f(a);
        }
        return next_ ? next_->forward_propagation(output_) : output_;
    }

    virtual const vec_t& back_propagation(const vec_t& current_delta, updater *l) {
        const vec_t& prev_out = prev_->output();
        const activation& prev_h = prev_->activation_function();

        if (l) {
            for (size_t i = 0; i < W_.size(); i++) {
                const io_connections& connections = weight2io_[i];
                float_t diff = 0.0;

                for (auto connection : connections)
                    diff += prev_out[connection.first] * current_delta[connection.second];

                diff *= scale_factor_;
                l->update(diff, Whessian_[i], &W_[i]);
            }

            for (size_t i = 0; i < b_.size(); i++) {
                std::vector<int>& outs = bias2out_[i];
                float_t diff = 0.0;

                for (auto o : outs)
                    diff += current_delta[o];    

                l->update(diff, bhessian_[i], &b_[i]);
            }
        }

        for (int i = 0; i < in_size_; i++) {
            const wo_connections& connections = in2wo_[i];
            prev_delta_[i] = 0.0;

            for (auto connection : connections) 
                prev_delta_[i] += W_[connection.first] * current_delta[connection.second];

            prev_delta_[i] *= prev_h.df(prev_out[i]);
        }
        return prev_->back_propagation(prev_delta_, l);
    }

    const vec_t& back_propagation_2nd(const vec_t& current_delta2) {
        const vec_t& prev_out = prev_->output();
        const activation& prev_h = prev_->activation_function();

        for (size_t i = 0; i < W_.size(); i++) {
            const io_connections& connections = weight2io_[i];
            float_t diff = 0.0;

            for (auto connection : connections)
                diff += prev_out[connection.first] * prev_out[connection.first] * current_delta2[connection.second];

            diff *= scale_factor_;
            Whessian_[i] += diff;
        }

        for (size_t i = 0; i < b_.size(); i++) {
            std::vector<int>& outs = bias2out_[i];
            float_t diff = 0.0;

            for (auto o : outs)
                diff += current_delta2[o];    

            bhessian_[i] += diff;
        }

        for (int i = 0; i < in_size_; i++) {
            const wo_connections& connections = in2wo_[i];
            prev_delta2_[i] = 0.0;

            for (auto connection : connections) 
                prev_delta2_[i] += W_[connection.first] * W_[connection.first] * current_delta2[connection.second];

            prev_delta2_[i] *= prev_h.df(prev_out[i]) * prev_h.df(prev_out[i]);
        }
        return prev_->back_propagation_2nd(prev_delta2_);
    }


protected:
    float_t scale_factor_;
    std::vector<wo_connections> in2wo_; // in_id -> [(weight_id, out_id)]
    std::vector<io_connections> weight2io_; // weight_id -> [(in_id, out_id)]
    std::vector<wi_connections> out2wi_; // out_id -> [(weight_id, in_id)]
    std::vector<std::vector<int> > bias2out_;
    std::vector<int> out2bias_;
};

} 