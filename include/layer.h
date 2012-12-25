#pragma once
#include "util.h"
#include "activation.h"
#include "updater.h"

namespace nn {

class layers;

// base class of all kind of NN layers
class layer_base {
public:
    layer_base(int in_dim, int out_dim, int weight_dim, int bias_dim) : next_(0), prev_(0) {
        set_size(in_dim, out_dim, weight_dim, bias_dim);
        initialize();
    }

    void connect(layer_base* tail) {
        if (this->out_size() != 0 && tail->in_size() != this->out_size())
            throw nn_error("dimension mismatch");
        next_ = tail;
        tail->prev_ = this;
    }

    void initialize() {
        uniform_rand(W_.begin(), W_.end(), -0.2, 0.2); // @todo —”‚Ì”ÍˆÍ‚ğÅ“K‰»‚·‚é
        uniform_rand(b_.begin(), b_.end(), -0.2, 0.2);       
    }

    vec_t& output() { return output_; }
    vec_t& delta() { return prev_delta_; }
    vec_t& weight() { return W_; }
    vec_t& bias() { return b_; }

    virtual int in_size() const { return in_size_; }
    virtual int out_size() const { return out_size_; }
    virtual int param_size() const { return W_.size() + b_.size(); }
    virtual int connection_size() const = 0;
    virtual void reset() { initialize(); }

    virtual activation& activation_function() = 0;
    virtual const vec_t& forward_propagation(const vec_t& in) = 0;
    virtual const vec_t& back_propagation(const vec_t& current_delta, updater *l) = 0;

protected:
    int in_size_;
    int out_size_;

    friend class layers;
    layer_base* next_;
    layer_base* prev_;
    vec_t output_;     // last output of current layer, set by fprop
    vec_t prev_delta_; // last delta of previous layer, set by bprop
    vec_t W_;          // weight vector
    vec_t b_;          // bias vector

private:
    void set_size(int in_dim, int out_dim, int weight_dim, int bias_dim) {
        in_size_ = in_dim;
        out_size_ = out_dim;
        output_.resize(out_dim);
        prev_delta_.resize(in_dim);
        W_.resize(weight_dim);
        b_.resize(bias_dim);     
    }
};

template<typename Activation>
class layer : public layer_base {
public:
    layer(int in_dim, int out_dim, int weight_dim, int bias_dim)
        : layer_base(in_dim, out_dim, weight_dim, bias_dim) {}

    activation& activation_function() { return a_; }

protected:
    Activation a_;
};


class input_layer : public layer<identity_activation> {
public:
    input_layer() : layer<identity_activation>(0, 0, 0, 0) {}

    const vec_t& forward_propagation(const vec_t& in) {
        output_ = in;
        return next_ ? next_->forward_propagation(in) : output_;
    }

    const vec_t& back_propagation(const vec_t& current_delta, updater *l) {
        return current_delta;
    }

    int connection_size() const {
        return in_size_;
    }
};

class layers {
public:
    layers() : head_(0), tail_(0) {
        add(&first_);
    }

    void add(layer_base * new_tail) {
        if (!head_) head_ = new_tail;
        if (tail_)  tail_->connect(new_tail);
        tail_ = new_tail;
    }
    bool empty() const { return head_ == 0; }
    layer_base* head() const { return head_; }
    layer_base* tail() const { return tail_; }
    void reset() {
        layer_base *l = head_;
        while(l) {
            l->reset();
            l = l->next_;
        }
    }
private:
    input_layer first_;
    layer_base *head_;
    layer_base *tail_;
};

}
