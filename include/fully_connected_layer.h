#pragma once
#include "cnn.h"
#include "learner.h"

namespace nn {

// normal 
class fully_connected_layer : public layer {
public:
    fully_connected_layer(int in_dim, int out_dim, activation *activation_function) : layer(in_dim, out_dim, in_dim * out_dim, out_dim) {
        set_activation(activation_function);
    }

    ~fully_connected_layer() {

    }

    int in_dim() const { return in_; }
    int out_dim() const { return out_; }
    int param_dim() const { return in_ * out_; }

    const vec_t& forward_propagation(const vec_t& in) {
        for (int r = 0; r < out_; r++) {
            float_t z = 0.0;
            for (int c = 0; c < in_; c++) 
                z += W_[r*in_+c] * in[c];
            z += b_[r];
            output_[r] = activation_->f(z);
        }

        return next_ ? next_->forward_propagation(output_) : output_;
    }

    const vec_t& back_propagation(const vec_t& current_delta, bool update) {
        const vec_t& prev_out = prev_->output();

        // update
        if (update) {
            for (int c = 0; c < in_; c++) 
                for (int r = 0; r < out_; r++)
                    learner_->update(current_delta[r] * prev_out[c], &W_[r*in_+c]); 
                     
            for (int r = 0; r < out_; r++)
                learner_->update(current_delta[r], &b_[r]);       
        }



        for (int c = 0; c < in_; c++) { 
            prev_delta_[c] = 0.0;

            for (int r = 0; r < out_; r++) 
                prev_delta_[c] += current_delta[r] * prev_out[c];

            prev_delta_[c] *= activation_->f(prev_out[c]);
        }

        return prev_->back_propagation(prev_delta_, update);
    }

private:

};

}
