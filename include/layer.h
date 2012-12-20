#pragma once
#include "util.h"

namespace nn {

class layers;

// base class of all kind of NN layers
class layer {
public:
    layer(int in_dim, int out_dim, int weight_dim, int bias_dim) : next_(0), prev_(0), activation_(0), learner_(0) {
        set_size(in_dim, out_dim, weight_dim, bias_dim);
        initialize();
    }

    void connect(layer* tail) {
        if (this->out_dim() != 0 && tail->in_dim() != this->out_dim())
            throw std::domain_error("dimension mismatch");
        next_ = tail;
        tail->prev_ = this;
    }

    void set_activation(activation *a) {
        activation_ = a;
    }

    void set_learner(learner *l) {
        learner_ = l;
    }

    void initialize() {
        uniform_rand(W_.begin(), W_.end(), -0.2, 0.2); // @todo —”‚Ì”ÍˆÍ‚ğÅ“K‰»‚·‚é
        uniform_rand(b_.begin(), b_.end(), -0.2, 0.2);       
    }

    vec_t& output() { return output_; }
    vec_t& delta() { return prev_delta_; }
    activation& activation_function() { return *activation_; }

    virtual int in_dim() const { return in_; }
    virtual int out_dim() const { return out_; }
    virtual void reset() { initialize(); }

    virtual const vec_t& forward_propagation(const vec_t& in) = 0;
    virtual const vec_t& back_propagation(const vec_t& current_delta, bool update = true) = 0;

protected:
    int in_;
    int out_;
    activation *activation_;
    learner *learner_;
    friend class layers;
    layer* next_;
    layer* prev_;
    vec_t output_;
    vec_t prev_delta_;
    vec_t W_;
    vec_t b_;

private:
    void set_size(int in_dim, int out_dim, int weight_dim, int bias_dim) {
        in_ = in_dim;
        out_ = out_dim;
        output_.resize(out_dim);
        prev_delta_.resize(in_dim);
        W_.resize(weight_dim);
        b_.resize(bias_dim);     
    }
};

class input_layer : public layer {
public:
    input_layer() : layer(0, 0, 0, 0) {}

    const vec_t& forward_propagation(const vec_t& in) {
        output_ = in;
        return next_ ? next_->forward_propagation(in) : output_;
    }

    const vec_t& back_propagation(const vec_t& current_delta, bool update = true) {
        return current_delta;
    }

};

class layers {
public:
    layers() : head_(0), tail_(0) {
        add(&first_);
    }

    void add(layer * new_tail) {
        if (!head_) head_ = new_tail;
        if (tail_)  tail_->connect(new_tail);
        tail_ = new_tail;
    }
    bool empty() const { return head_ == 0; }
    layer* head() const { return head_; }
    layer* tail() const { return tail_; }
    void reset() {
        layer *l = head_;
        while(l) {
            l->reset();
            l = l->next_;
        }
    }
private:
    input_layer first_;
    layer *head_;
    layer *tail_;
};

}
