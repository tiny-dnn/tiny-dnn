#pragma once
#include "util.h"
#include "partial_connected_layer.h"

namespace tiny_cnn {

struct connection_table {
    connection_table() : rows_(0), cols_(0) {}
    connection_table(const bool *ar, size_t rows, size_t cols) : connected_(rows * cols), rows_(rows), cols_(cols) {
        std::copy(ar, ar + rows * cols, connected_.begin());
    }

    bool is_connected(int x, int y) const {
        return is_empty() ? true : connected_[y * cols_ + x];
    }

    bool is_empty() const {
        return rows_ == 0 && cols_ == 0;
    }

    std::vector<bool> connected_;
    size_t rows_;
    size_t cols_;
};

template<typename N, typename Activation>
class convolutional_layer : public partial_connected_layer<N, Activation> {
public:
    typedef partial_connected_layer<N, Activation> Base;
    typedef typename Base::Updater Updater;

    convolutional_layer(int in_width, int in_height, int window_size, int in_channels, int out_channels)
    : partial_connected_layer<N, Activation>(in_width * in_height * in_channels, (in_width - window_size + 1) * (in_height - window_size + 1) * out_channels, 
    window_size * window_size * in_channels * out_channels, out_channels), 
    in_(in_width, in_height, in_channels), 
    out_((in_width - window_size + 1), (in_height - window_size + 1), out_channels),
    weight_(window_size, window_size, in_channels*out_channels),
    window_size_(window_size)
    {
        init_connection(connection_table());
    }

    convolutional_layer(int in_width, int in_height, int window_size, int in_channels, int out_channels, const connection_table& connection_table)
        : partial_connected_layer<N, Activation>(in_width * in_height * in_channels, (in_width - window_size + 1) * (in_height - window_size + 1) * out_channels, 
        window_size * window_size * in_channels * out_channels, out_channels), 
        in_(in_width, in_height, in_channels), 
        out_((in_width - window_size + 1), (in_height - window_size + 1), out_channels),
        weight_(window_size, window_size, in_channels*out_channels),
        window_size_(window_size)
    {
        init_connection(connection_table);
        this->remap();
    }

private:
    void init_connection(const connection_table& table) {
        int n = 0;
        for (int inc = 0; inc < in_.depth_; inc++) {
            for (int outc = 0; outc < out_.depth_; outc++) {
                if (!table.is_connected(outc, inc)) {
                    n++;
                    continue;
                }

                for (int y = 0; y < out_.height_; y++)
                    for (int x = 0; x < out_.width_; x++)
                        connect_kernel(inc, outc, x, y);
            }
        }

        for (int outc = 0; outc < out_.depth_; outc++)
            for (int y = 0; y < out_.height_; y++)
                for (int x = 0; x < out_.width_; x++)
                    this->connect_bias(outc, out_.get_index(x, y, outc));
    }

    void connect_kernel(int inc, int outc, int x, int y) {
        for (int dy = 0; dy < window_size_; dy++)
            for (int dx = 0; dx < window_size_; dx++)
                this->connect_weight(
                    in_.get_index(x + dx, y + dy, inc), 
                    out_.get_index(x, y, outc), 
                    weight_.get_index(dx, dy, outc * in_.depth_ + inc));
    }

    tensor3d in_;
    tensor3d out_;
    tensor3d weight_;
    int window_size_;
};

} 