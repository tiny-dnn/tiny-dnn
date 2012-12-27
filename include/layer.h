#pragma once
#include "util.h"
#include "activation.h"
#include "updater.h"

namespace nn {

template<typename N>
class layers;

// base class of all kind of NN layers
template<typename N>
class layer_base {
public:
    typedef N Network;
    typedef typename Network::Updater Updater;
    typedef typename Network::LossFunction LossFunction;

    layer_base(){}

    layer_base(int in_dim, int out_dim, int weight_dim, int bias_dim) : next_(0), prev_(0) {
        set_size(in_dim, out_dim, weight_dim, bias_dim);
    }


    void connect(layer_base<N>* tail) {
        if (this->out_size() != 0 && tail->in_size() != this->out_size())
            throw nn_error("dimension mismatch");
        next_ = tail;
        tail->prev_ = this;
    }

    void init_weight() {
        const float_t weight_base = 3.0 / std::sqrt(fan_in_size());

        uniform_rand(W_.begin(), W_.end(), -weight_base, weight_base);
        uniform_rand(b_.begin(), b_.end(), -weight_base, weight_base);               
        std::fill(Whessian_.begin(), Whessian_.end(), 0.0);
        std::fill(bhessian_.begin(), bhessian_.end(), 0.0);
    }

    vec_t& output() { return output_; }
    vec_t& delta() { return prev_delta_; }
    vec_t& weight() { return W_; }
    vec_t& bias() { return b_; }

    void divide_hessian(int denominator) { 
        for (size_t i = 0; i < Whessian_.size(); i++) Whessian_[i] /= denominator;
        for (size_t i = 0; i < bhessian_.size(); i++) bhessian_[i] /= denominator;
    }

    virtual int in_size() const { return in_size_; }
    virtual int out_size() const { return out_size_; }
    virtual int param_size() const { return W_.size() + b_.size(); }
    virtual int fan_in_size() const = 0;
    virtual int connection_size() const = 0;
    virtual void reset() { init_weight(); }

    virtual activation& activation_function() = 0;
    virtual const vec_t& forward_propagation(const vec_t& in) = 0;
    virtual const vec_t& back_propagation(const vec_t& current_delta, Updater *l) = 0;
    virtual const vec_t& back_propagation_2nd(const vec_t& current_delta2) = 0;

    layer_base<N>* next() { return next_; }
    layer_base<N>* prev() { return prev_; }

protected:
    int in_size_;
    int out_size_;

    layer_base<N>* next_;
    layer_base<N>* prev_;
    vec_t output_;     // last output of current layer, set by fprop
    vec_t prev_delta_; // last delta of previous layer, set by bprop
    vec_t W_;          // weight vector
    vec_t b_;          // bias vector

    vec_t Whessian_; // diagonal terms of hessian matrix
    vec_t bhessian_;
    vec_t prev_delta2_; // d^2E/da^2

private:
    void set_size(int in_dim, int out_dim, int weight_dim, int bias_dim) {
        in_size_ = in_dim;
        out_size_ = out_dim;
        output_.resize(out_dim);
        prev_delta_.resize(in_dim);
        W_.resize(weight_dim);
        b_.resize(bias_dim);     
        Whessian_.resize(weight_dim);
        bhessian_.resize(bias_dim);
        prev_delta2_.resize(in_dim);
    }
};

template<typename N, typename Activation>
class layer : public layer_base<N> {
public:
    typedef layer_base<N> Base;
    typedef typename Base::Updater Updater;

    layer(int in_dim, int out_dim, int weight_dim, int bias_dim)
        : layer_base<N>(in_dim, out_dim, weight_dim, bias_dim) {}

    activation& activation_function() { return a_; }

protected:
    Activation a_;
};

template<typename N>
class input_layer : public layer<N, identity_activation> {
public:
    typedef layer<N, identity_activation> Base;
    typedef typename Base::Updater Updater;

    input_layer() : layer<N, identity_activation>(0, 0, 0, 0) {}

    const vec_t& forward_propagation(const vec_t& in) {
        this->output_ = in;
        return this->next_ ? this->next_->forward_propagation(in) : this->output_;
    }

    const vec_t& back_propagation(const vec_t& current_delta, Updater *l) {
        return current_delta;
    }

    const vec_t& back_propagation_2nd(const vec_t& current_delta2) {
        return current_delta2;
    }

    int connection_size() const {
        return this->in_size_;
    }

    int fan_in_size() const {
        return 1;
    }
};


template<typename U>
class layers {
public:
    layers() : head_(0), tail_(0) {
        add(&first_);
    }

    void add(layer_base<U> * new_tail) {
        if (!head_) head_ = new_tail;
        if (tail_)  tail_->connect(new_tail);
        tail_ = new_tail;
    }
    bool empty() const { return head_ == 0; }
    layer_base<U>* head() const { return head_; }
    layer_base<U>* tail() const { return tail_; }
    void reset() {
        layer_base<U> *l = head_;
        while(l) {
            l->reset();
            l = l->next();
        }
    }
    void divide_hessian(int denominator) {
        layer_base<U> *l = head_;
        while(l) {
            l->divide_hessian(denominator);
            l = l->next();
        }     
    }
private:
    input_layer<U> first_;
    layer_base<U> *head_;
    layer_base<U> *tail_;
};

}
