#pragma once
#include "cnn.h"

namespace nn {

// normal 
template<int In, int Out, typename Activation = sigmoid_activation>
class fully_connected_layer : public layer {
public:
    fully_connected_layer() : layer(In, Out, In*Out+Out){
        W_.resize(In*Out);
        b_.resize(Out);
        dW_.resize(In*Out);
        dB_.resize(Out);
    }

    int in_dim() const { return In; }
    int out_dim() const { return Out; }
    int param_dim() const { return In*Out+Out; }

    const vec_t* forward_propagation(const vec_t& in) {
        for (int r = 0; r < Out; r++) {
            float_t z = 0.0;
            for (int c = 0; c < In; c++) 
                z += W_[r*In+c] * in[c];
            z += b_[r];
            output_[r] = Activation::f(z);
        }

        return next_ ? next_->forward_propagation(output_) : &output_;
    }

    const vec_t* back_propagation(const vec_t& in, const vec_t& train_signal) {
        if (!next_) {
            for (int i = 0; i < Out; i++)
                delta_[i] = (output_[i] - train_signal[i]) * Activation::df(output_[i]);      
        }

        const vec_t& prev_out = prev_ ? prev_->output() : in;
        for (int c = 0; c < In; c++) 
            for (int r = 0; r < Out; r++)
                dW_[r*In+c] += delta_[r] * prev_out[c];
        for (int r = 0; r < Out; r++)
            dB_[r] += delta_[r];

        if (!prev_) return &delta_;

        for (int c = 0; c < In; c++) {
            prev_->delta()[c] = 0.0;
            for (int r = 0; r < Out; r++)
                prev_->delta()[c] += delta_[r] * W_[r*In+c];
            prev_->delta()[c] *= Activation::df(prev_->output()[c]);
        }

        return prev_->back_propagation(in, train_signal);
    }

    void unroll(pvec_t *w, pvec_t *dw, pvec_t *b, pvec_t *db) {
        for (size_t i = 0; i < W_.size(); i++)
            w->push_back(&W_[i]);
        for (size_t i = 0; i < b_.size(); i++)
            b->push_back(&b_[i]);
        for (size_t i = 0; i < dW_.size(); i++)
            dw->push_back(&dW_[i]);
        for (size_t i = 0; i < dB_.size(); i++)
            db->push_back(&dB_[i]);
    }

private:
    vec_t dW_;
    vec_t dB_;
    vec_t W_;
    vec_t b_;
};

}
