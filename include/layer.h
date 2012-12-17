#pragma once
#include "util.h"

namespace nn {

class layers;

// base class of all kind of NN layers
class layer {
public:
    layer(int in, int out, int param) : next_(0), prev_(0) {
        output_.resize(out);
        delta_.resize(out);
    }
    void connect(layer* tail) {
        if (tail->in_dim() != this->out_dim())
            throw std::domain_error("dimension mismatch");
        next_ = tail;
        tail->prev_ = this;
    }
    vec_t& output() { return output_; }
    vec_t& delta() { return delta_; }

    virtual int in_dim() const = 0; // 入力次元
    virtual int out_dim() const = 0; // 出力次元
    virtual int param_dim() const = 0; // パラメータ次元

    // in: 前層の出力。in.size() == in_dim()
    // ret: 出力層の出力ベクトル。
    virtual const vec_t* forward_propagation(const vec_t& in) = 0;

    // in: 出力層の出力ベクトル。 train_signal: 教師ベクトル
    // ret: 入力層のδ
    virtual const vec_t* back_propagation(const vec_t& in, const vec_t& train_signal) = 0;

    virtual void unroll(pvec_t *w, pvec_t *dw, pvec_t *b, pvec_t *db) = 0;

protected:
    friend class layers;
    layer* next_;
    layer* prev_;
    vec_t output_;
    vec_t delta_;
};

class layers {
public:
    layers() : head_(0), tail_(0) {}
    void add(layer * new_tail) {
        if (!head_) head_ = new_tail;
        if (tail_)  tail_->connect(new_tail);
        tail_ = new_tail;

        new_tail->unroll(&w_, &dw_, &b_, &db_);
        unroll();
    }
    bool empty() const { return head_ == 0; }
    layer* head() const { return head_; }
    layer* tail() const { return tail_; }
    pvec_t& all_param() { return wb_; }
    pvec_t& all_diff() { return dwb_; }
    pvec_t& weight() { return w_; }
    pvec_t& bias() { return b_; }
    pvec_t& weight_diff() { return dw_; }
    pvec_t& bias_diff() { return db_; }
    void reset_diff() {
        for (auto d : dwb_)
            *d = 0.0;
    }
private:
    void unroll() {
        wb_ = w_;
        std::copy(b_.begin(), b_.end(), std::back_inserter(wb_));
        dwb_ = dw_;
        std::copy(db_.begin(), db_.end(), std::back_inserter(dwb_));
    }
    layer *head_;
    layer *tail_;
    pvec_t w_;
    pvec_t b_;
    pvec_t wb_;
    pvec_t dw_;
    pvec_t db_;
    pvec_t dwb_;
};

}
