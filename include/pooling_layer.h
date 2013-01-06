#pragma once
#include "util.h"
#include "partial_connected_layer.h"

namespace tiny_cnn {


template<typename N, typename Activation>
class average_pooling_layer : public partial_connected_layer<N, Activation> {
public:
    typedef partial_connected_layer<N, Activation> Base;
    typedef typename Base::Updater Updater;

    average_pooling_layer(int in_width, int in_height, int in_channels, int pooling_size)
    : partial_connected_layer<N, Activation>(
     in_width * in_height * in_channels, 
     in_width * in_height * in_channels / (pooling_size * pooling_size), 
     in_channels, in_channels, 1.0 / (pooling_size * pooling_size)),
     in_(in_width, in_height, in_channels), 
     out_(in_width/pooling_size, in_height/pooling_size, in_channels)
    {
        if ((in_width % pooling_size) || (in_height % pooling_size)) 
            throw nn_error("width/height must be multiples of pooling size");
        init_connection(pooling_size);
    }

private:
    void init_connection(int pooling_size) {
        for (int c = 0; c < in_.depth_; c++) 
            for (int y = 0; y < in_.height_; y += pooling_size)
                for (int x = 0; x < in_.width_; x += pooling_size)
                    connect_kernel(pooling_size, x, y, c);


        for (int c = 0; c < in_.depth_; c++) 
            for (int y = 0; y < out_.height_; y++)
                for (int x = 0; x < out_.width_; x++)
                    this->connect_bias(c, out_.get_index(x, y, c));
    }

    void connect_kernel(int pooling_size, int x, int y, int inc) {
        for (int dy = 0; dy < pooling_size; dy++)
            for (int dx = 0; dx < pooling_size; dx++)
                this->connect_weight(
                    in_.get_index(x + dx, y + dy, inc), 
                    out_.get_index(x / pooling_size, y / pooling_size, inc),
                    inc);
    }

    tensor3d in_;
    tensor3d out_;
};

} 