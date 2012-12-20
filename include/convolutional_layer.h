#pragma once
#include "util.h"
#include "layer.h"

namespace nn {

class partial_connected_layer : public layer {
public:
    typedef std::vector<std::pair<int, int> > io_connections;
    typedef std::vector<std::pair<int, int> > wi_connections;

    partial_connected_layer(int in_dim, int out_dim, int weight_dim, int bias_dim)
        : layer (in_dim, out_dim, weight_dim, bias_dim), weight2io_(weight_dim), out2wi_(out_dim), bias2out_(bias_dim), out2bias_(out_dim) {}

    void connect_weight(int input_index, int output_index, int weight_index) {
        weight2io_[weight_index].push_back(std::make_pair(input_index, output_index));
        out2wi_[output_index].push_back(std::make_pair(weight_index, input_index));
    }

    void connect_bias(int bias_index, int output_index) {
        out2bias_[output_index] = bias_index;
        bias2out_[bias_index].push_back(output_index);
    }

    const vec_t& forward_propagation(const vec_t& in) {
    // TODO
        return next_ ? next_->forward_propagation(output_) : output_;
    }

    const vec_t& back_propagation(const vec_t& current_delta, bool update) {
    // TODO
        return prev_->back_propagation(prev_delta_, update);
    }

private:
    std::vector<io_connections> weight2io_; // weight_id -> [(in_id, out_id)]
    std::vector<wi_connections> out2wi_; // out_id -> [(weight_id, in_id)]
    std::vector<std::vector<int> > bias2out_;
    std::vector<int> out2bias_;
};


class convolutional_layer : public partial_connected_layer {
public:
    convolutional_layer(int in_width, int in_height, int window_size, int in_channels, int out_channels)
    : partial_connected_layer(in_width * in_height * in_channels, (in_width - window_size + 1) * (in_height - window_size + 1) * out_channels, 
    window_size * window_size, out_channels), 
    in_(in_width, in_height, in_channels), 
    out_((in_width - window_size + 1), (in_height - window_size + 1), out_channels),
    window_size_(window_size)
    {
        connect();
    }


private:
    struct convolutional_structure {
        convolutional_structure(int width, int height, int channels) : width_(width), height_(height), channels_(channels) {}

        int get_index(int x, int y, int channel) const {
            return (width_ * height_) * channel + width_ * y + x;
        }
        int width_;
        int height_;
        int channels_;
    };

    void connect() {
        for (int inc = 0; inc < in_.channels_; inc++)
            for (int outc = 0; outc < out_.channels_; outc++)
                for (int y = 0; y < out_.height_; y++)
                    for (int x = 0; x < out_.width_; x++)
                        connect_kernel(inc, outc, x, y);

        for (int outc = 0; outc < out_.channels_; outc++)
            for (int y = 0; y < out_.height_; y++)
                for (int x = 0; x < out_.width_; x++)
                    connect_bias(outc, out_.get_index(x, y, outc));
    }

    void connect_kernel(int inc, int outc, int x, int y) {
        for (int dy = 0; dy < window_size_; dy++)
            for (int dx = 0; dx < window_size_; dx++)
                connect_weight(in_.get_index(x + dx, y + dy, inc), out_.get_index(x, y, outc), dy * window_size_ + dx);
    }

    int window_size_;
    convolutional_structure in_;
    convolutional_structure out_;
};

} 