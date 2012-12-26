#pragma once
#include "layer.h"
#include "updater.h"

namespace nn {

// normal 
template<typename Activation>
class fully_connected_layer : public layer<Activation> {
public:
    fully_connected_layer(int in_dim, int out_dim) : layer<Activation>(in_dim, out_dim, in_dim * out_dim, out_dim) {}

    int connection_size() const {
        return in_size_ * out_size_ + out_size_;
    }

    int fan_in_size() const {
        return in_size_;
    }

    const vec_t& forward_propagation(const vec_t& in) {
        for (int r = 0; r < out_size_; r++) {
            float_t z = 0.0;
            for (int c = 0; c < in_size_; c++) 
                z += W_[r*in_size_+c] * in[c];
            z += b_[r];
            output_[r] = a_.f(z);
        }

        return next_ ? next_->forward_propagation(output_) : output_;
    }

    const vec_t& back_propagation(const vec_t& current_delta, updater *l) {
        const vec_t& prev_out = prev_->output();
        const activation& prev_h = prev_->activation_function();

        if (l) {
            for (int r = 0; r < out_size_; r++)
                for (int c = 0; c < in_size_; c++) 
                    l->update(current_delta[r] * prev_out[c], Whessian_[r*in_size_+c], &W_[r*in_size_+c]); 
                     
            for (int r = 0; r < out_size_; r++)
                l->update(current_delta[r], bhessian_[r], &b_[r]);   
        }

        for (int c = 0; c < in_size_; c++) { 
            prev_delta_[c] = 0.0;

            for (int r = 0; r < out_size_; r++) 
                prev_delta_[c] += current_delta[r] * W_[r*in_size_+c];

            prev_delta_[c] *= prev_h.df(prev_out[c]);
        }

        return prev_->back_propagation(prev_delta_, l);
    }


    const vec_t& back_propagation_2nd(const vec_t& current_delta2) {
        const vec_t& prev_out = prev_->output();
        const activation& prev_h = prev_->activation_function();

        for (int r = 0; r < out_size_; r++)
            for (int c = 0; c < in_size_; c++) 
                Whessian_[r*in_size_+c] += current_delta2[r] * prev_out[c] * prev_out[c];

        for (int r = 0; r < out_size_; r++)
            bhessian_[r] += current_delta2[r];

        for (int c = 0; c < in_size_; c++) { 
            prev_delta2_[c] = 0.0;

            for (int r = 0; r < out_size_; r++) 
                prev_delta2_[c] += current_delta2[r] * W_[r*in_size_+c] * W_[r*in_size_+c];

            prev_delta2_[c] *= prev_h.df(prev_out[c]) * prev_h.df(prev_out[c]);
        }

        return prev_->back_propagation_2nd(prev_delta2_);
    }
};

}
