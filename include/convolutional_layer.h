#pragma once
#include "util.h"
#include "layer.h"

namespace nn {

struct convolutional_prop {
    convolutional_prop(int ch, int w, int h) : channels(ch), width(w), height(h) {}
    int channels;
    int width;
    int height;
    size_t size() const { return channels * width * height; }
};

template<typename T>
class vector3d {
public:
    vector3d(int width, int height, int depth)
     : width_(width), height_(height), depth_(depth), vec_(width * height * depth){}

    vector3d(T vec, int width, int height, int depth)
        : width_(width), height_(height), depth_(depth), vec_(vec){ 
        if(vec_.size() != size()) 
            throw std::domain_error("dimension mismatch");
    }

    vector3d(const convolutional_prop& param)
        : width_(param.width), height_(param.height), depth_(param.channels), vec_(param.size()){}

    size_t size() const {
        return width_ * height_ * depth_;
    }

    float_t& at(int x, int y, int z) {
        return vec_[z * (width_ * height_) + y * (width_) + x];
    }
    const float_t& at(int x, int y, int z) const {
        return vec_[z * (width_ * height_) + y * (width_) + x];
    }
    float_t& operator [] (size_t index) {
        return vec_[index];
    }
    const float_t& operator [] (size_t index) const {
        return vec_[index];
    }
private:
    int width_;
    int height_;
    int depth_;
    T vec_;
};

template<typename Activation = sigmoid_activation>
class convolutional_layer : public layer{
public:
    convolutional_layer(const convolutional_prop& in, const convolutional_prop& out, int window_size)
     : layer(in.size(), out.size(), window_size * window_size * in.channels * out.channels),
       in_(in), out_(out), window_size_(window_size), b_(out.channels), db_(out.channels), 
       W_(in.channels, vector3d<vec_t>(window_size, window_size, out.channels)), 
       dW_(in.channels, vector3d<vec_t>(window_size, window_size, out.channels)),
       output_table_(output_, out.width, out.height, out.channels) {

    }

    int in_dim() const { return in_.size(); }
    int out_dim() const { return out_.size(); }
    int param_dim() const { return window_size_ * window_size_ * in_.channels * out_.channels; }

    const vec_t* forward_propagation(const vec_t& in) {
        vector3d<const vec_t&> in_vec(in, in_.width, in_.height, in_.channels);

        for (int outc = 0; outc < out_.channels; outc++)
            for (int inc = 0; inc < in_.channels; inc++) 
                convolute(in_vec, W_[inc], inc, outc, &output_table_);

        return 0; // TODO
    }
    const vec_t* back_propagation(const vec_t& in, const vec_t& train_signal) {
        if (!next_) {
            //for (int i = 0; i < Out; i++)
            //    delta_[i] = (output_[i] - train_signal[i]) * Activation::df(output_[i]);      
        }

        const vec_t& prev_out = prev_ ? prev_->output() : in;
        /*for (int c = 0; c < In; c++) 
            for (int r = 0; r < Out; r++)
                dW_[r*In+c] += delta_[r] * prev_out[c];
        for (int r = 0; r < Out; r++)
            dB_[r] += delta_[r];*/

        if (!prev_) return &delta_;

        /*for (int c = 0; c < In; c++) {
            prev_->delta()[c] = 0.0;
            for (int r = 0; r < Out; r++)
                prev_->delta()[c] += delta_[r] * W_[r*In+c];
            prev_->delta()[c] *= Activation::df(prev_->output()[c]);
        }*/

        return prev_->back_propagation(in, train_signal);
    }
    void unroll(pvec_t *w, pvec_t *dw, pvec_t *b, pvec_t *db) {
        for (auto v: W_) {
            for (size_t i = 0; i < v.size(); i++)
                w->push_back(&v[i]);
        }
        for (auto v: dW_) {
            for (int i = 0; i < v.size(); i++)
                dw->push_back(&v[i]);
        }
        for (auto v: b_)
            b->push_back(&v);
        for (auto v: db_)
            db->push_back(&v);
    }

private:
    void convolute(const vector3d<const vec_t&>& in, const vector3d<vec_t>& W, int inc, int outc, vector3d<vec_t&> *output) {
        for (int y = 0; y < out_.height; y++) {
            for (int x = 0; x < out_.width; x++) {
                float_t val = 0.0;

                for (int dy = 0; dy < window_size_; dy++)
                    for (int dx = 0; dx < window_size_; dx++)
                        val += W.at(dx, dy, outc) * in.at(x + dx, y + dy, inc);
                val += b_[outc];
                output->at(x, y, outc) = Activation::f(val);
            }
        }
    }

    convolutional_prop in_;
    convolutional_prop out_;
    int window_size_;

    std::vector<vector3d<vec_t> > W_;
    std::vector<vector3d<vec_t> > dW_;
    vec_t b_;
    vec_t db_;
    vector3d<vec_t&> output_table_;
};

} 