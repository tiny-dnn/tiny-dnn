#pragma once
#include "util.h"
#include "partial_connected_layer.h"

namespace nn {

template<int inc, int outc>
struct convolutional_connection {

};

template<typename Activation>
class convolutional_layer : public partial_connected_layer<Activation> {
public:
    convolutional_layer(int in_width, int in_height, int window_size, int in_channels, int out_channels)
    : partial_connected_layer<Activation>(in_width * in_height * in_channels, (in_width - window_size + 1) * (in_height - window_size + 1) * out_channels, 
    window_size * window_size * in_channels * out_channels, out_channels), 
    in_(in_width, in_height, in_channels), 
    out_((in_width - window_size + 1), (in_height - window_size + 1), out_channels),
    weight_(window_size, window_size, in_channels*out_channels),
    window_size_(window_size)
    {
        init_connection();
    }

private:
    void init_connection() {
        for (int inc = 0; inc < in_.depth_; inc++)
            for (int outc = 0; outc < out_.depth_; outc++)
                for (int y = 0; y < out_.height_; y++)
                    for (int x = 0; x < out_.width_; x++)
                        connect_kernel(inc, outc, x, y);

        for (int outc = 0; outc < out_.depth_; outc++)
            for (int y = 0; y < out_.height_; y++)
                for (int x = 0; x < out_.width_; x++)
                    connect_bias(outc, out_.get_index(x, y, outc));
    }

    void connect_kernel(int inc, int outc, int x, int y) {
        for (int dy = 0; dy < window_size_; dy++)
            for (int dx = 0; dx < window_size_; dx++)
                connect_weight(
                    in_.get_index(x + dx, y + dy, inc), 
                    out_.get_index(x, y, outc), 
                    weight_.get_index(dx, dy, outc * in_.depth_ + inc));
    }

    int window_size_;
    tensor3d in_;
    tensor3d out_;
    tensor3d weight_;
};

} 