#pragma once
#include "util.h"
#include "layer.h"

namespace nn {

class convolutional_layer : public layer{
public:
    convolutional_layer(int in_channels, int out_channels, int window_size, int in_width, int in_height)
     : layer(in_width * in_height * in_channels, 
             (in_width - window_size + 1) * (in_height - window_size + 1) * out_channels, 
             window_size * window_size * in_channels * out_channels),
       in_width_(in_width), in_height_(in_height), in_channels_(in_channels), out_channels_(out_channels), window_size_(window_size) {

    }

    int in_dim() const { return in_width_ * in_height_ * in_channels_; }
    int out_dim() const { return (in_width_ - window_size_ + 1) * (in_height_ - window_size_ + 1) * out_channels_; }
    int param_dim() const { return window_size_ * window_size_ * in_channels_ * out_channels_; }

    const vec_t* forward_propagation(const vec_t& in) {
        return 0; // TODO
    }
    const vec_t* back_propagation(const vec_t& in, const vec_t& train_signal) {
        return 0; // TODO
    }
    void unroll(pvec_t *w, pvec_t *dw, pvec_t *b, pvec_t *db) {
        // TODO
    }

private:
    int in_width_;
    int in_height_;
    int in_channels_;
    int out_channels_;
    int window_size_;
};

}