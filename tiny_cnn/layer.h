/*
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY 
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND 
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#include <sstream>
#include <iomanip>
#include "util.h"
#include "product.h"
#include "image.h"
#include "activation_function.h"

namespace tiny_cnn {


// base class of all kind of NN layers
class layer_base {
public:
    friend void connection_mismatch(const layer_base& from, const layer_base& to);

    virtual ~layer_base() {}

    layer_base(layer_size_t in_dim, layer_size_t out_dim, size_t weight_dim, size_t bias_dim) : parallelize_(true), next_(nullptr), prev_(nullptr) {
        set_size(in_dim, out_dim, weight_dim, bias_dim);
    }

    void connect(layer_base* tail) {
        if (out_size() != 0 && tail->in_size() != out_size())
            connection_mismatch(*this, *tail);
        next_ = tail;
        tail->prev_ = this;
    }

    void set_parallelize(bool parallelize) {
        parallelize_ = parallelize;
    }

    // cannot call from ctor because of pure virtual function call fan_in_size().
    // so should call this function explicitly after ctor
    void init_weight() {
        const float_t weight_base = 0.5 / std::sqrt(fan_in_size());

        uniform_rand(W_.begin(), W_.end(), -weight_base, weight_base);
        uniform_rand(b_.begin(), b_.end(), -weight_base, weight_base);               
        std::fill(Whessian_.begin(), Whessian_.end(), 0.0);
        std::fill(bhessian_.begin(), bhessian_.end(), 0.0);
        clear_diff(CNN_TASK_SIZE);
    }

    void divide_hessian(int denominator) {
        for (auto& w : Whessian_) w /= denominator;
        for (auto& b : bhessian_) b /= denominator;
    }

    // getter
    const vec_t& output(int worker_index) const { return output_[worker_index]; }
    const vec_t& delta(int worker_index) const { return prev_delta_[worker_index]; }
    vec_t& weight() { return W_; }
    vec_t& bias() { return b_; }
    vec_t& weight_diff(int index) { return dW_[index]; }
    vec_t& bias_diff(int index) { return db_[index]; }
    bool is_exploded() const { return has_infinite(W_) || has_infinite(b_); }
    layer_base* next() { return next_; }
    layer_base* prev() { return prev_; }

    virtual layer_size_t in_size() const { return in_size_; }
    virtual layer_size_t out_size() const { return out_size_; }
    virtual size_t param_size() const { return W_.size() + b_.size(); }
    virtual size_t fan_in_size() const = 0;
    virtual size_t connection_size() const = 0;
    virtual index3d<layer_size_t> in_shape() const { return index3d<layer_size_t>(in_size(), 1, 1); }
    virtual index3d<layer_size_t> out_shape() const { return index3d<layer_size_t>(out_size(), 1, 1); }
    virtual std::string layer_type() const = 0;

    virtual void save(std::ostream& os) const {
        if (is_exploded()) throw nn_error("failed to save weights because of infinite weight");
        for (auto w : W_) os << w << " ";
        for (auto b : b_) os << b << " ";
    }

    virtual void load(std::istream& is) {
        for (auto& w : W_) is >> w;
        for (auto& b : b_) is >> b;
    }

    virtual image output_to_image(size_t worker_index = 0) const {
        return vec2image(output_[worker_index]);
    }

    virtual activation::function& activation_function() = 0;
    virtual const vec_t& forward_propagation(const vec_t& in, size_t worker_index) = 0;
    virtual const vec_t& back_propagation(const vec_t& current_delta, size_t worker_index) = 0;
    virtual const vec_t& back_propagation_2nd(const vec_t& current_delta2) = 0;

    // called afrer updating weight
    virtual void post_update() {}

    template <typename Optimizer>
    void update_weight(Optimizer *o, int worker_size, size_t batch_size) {
        if (W_.empty()) return;

        merge(worker_size, batch_size);

        o->update(dW_[0], Whessian_, W_);
        o->update(db_[0], bhessian_, b_);

        clear_diff(worker_size);
        post_update();
    }

    bool has_same_weights(const layer_base& rhs, float_t eps) const {
        if (W_.size() != rhs.W_.size() || b_.size() != rhs.b_.size())
            return false;

        for (size_t i = 0; i < W_.size(); i++)
          if (std::abs(W_[i] - rhs.W_[i]) > eps) return false;
        for (size_t i = 0; i < b_.size(); i++)
          if (std::abs(b_[i] - rhs.b_[i]) > eps) return false;

        return true;
    }

protected:
    layer_size_t in_size_;
    layer_size_t out_size_;
    bool parallelize_;

    layer_base* next_;
    layer_base* prev_;
    vec_t a_[CNN_TASK_SIZE];          // w * x
    vec_t output_[CNN_TASK_SIZE];     // last output of current layer, set by fprop
    vec_t prev_delta_[CNN_TASK_SIZE]; // last delta of previous layer, set by bprop
    vec_t W_;          // weight vector
    vec_t b_;          // bias vector
    vec_t dW_[CNN_TASK_SIZE];
    vec_t db_[CNN_TASK_SIZE];

    vec_t Whessian_; // diagonal terms of hessian matrix
    vec_t bhessian_;
    vec_t prev_delta2_; // d^2E/da^2

private:
    void merge(size_t worker_size, size_t batch_size) {
        for (size_t i = 1; i < worker_size; i++)
            vectorize::reduce<float_t>(&dW_[i][0], dW_[i].size(), &dW_[0][0]);
        for (size_t i = 1; i < worker_size; i++)
            vectorize::reduce<float_t>(&db_[i][0], db_[i].size(), &db_[0][0]);

        std::transform(dW_[0].begin(), dW_[0].end(), dW_[0].begin(), [&](float_t x) { return x / batch_size; });
        std::transform(db_[0].begin(), db_[0].end(), db_[0].begin(), [&](float_t x) { return x / batch_size; });
    }

    void clear_diff(size_t worker_size) {
        for (size_t i = 0; i < worker_size; i++) {
            std::fill(dW_[i].begin(), dW_[i].end(), 0.0);
            std::fill(db_[i].begin(), db_[i].end(), 0.0);
        }
    }

    void set_size(layer_size_t in_dim, layer_size_t out_dim, size_t weight_dim, size_t bias_dim) {
        in_size_ = in_dim;
        out_size_ = out_dim;

        W_.resize(weight_dim);
        b_.resize(bias_dim);
        Whessian_.resize(weight_dim);
        bhessian_.resize(bias_dim);
        prev_delta2_.resize(in_dim);

        for (auto& o : output_)     o.resize(out_dim);
        for (auto& a : a_)          a.resize(out_dim);
        for (auto& p : prev_delta_) p.resize(in_dim);
        for (auto& dw : dW_) dw.resize(weight_dim);
        for (auto& db : db_) db.resize(bias_dim);
    }
};

template<typename Activation>
class layer : public layer_base {
public:
    layer(layer_size_t in_dim, layer_size_t out_dim, size_t weight_dim, size_t bias_dim)
        : layer_base(in_dim, out_dim, weight_dim, bias_dim) {}

    activation::function& activation_function() override { return h_; }
protected:
    Activation h_;
};

template <typename Char, typename CharTraits>
std::basic_ostream<Char, CharTraits>& operator << (std::basic_ostream<Char, CharTraits>& os, const layer_base& v) {
    v.save(os);
    return os;
}

template <typename Char, typename CharTraits>
std::basic_istream<Char, CharTraits>& operator >> (std::basic_istream<Char, CharTraits>& os, layer_base& v) {
    v.load(os);
    return os;
}

// error message functions

inline void connection_mismatch(const layer_base& from, const layer_base& to) {
    std::ostringstream os;

    os << std::endl;
    os << "output size of Nth layer must be equal to input of (N+1)th layer" << std::endl;
    os << "layerN:   " << std::setw(12) << from.layer_type() << " in:" << from.in_size() << "(" << from.in_shape() << "), " << 
                                                "out:" << from.out_size() << "(" << from.out_shape() << ")" << std::endl;
    os << "layerN+1: " << std::setw(12) << to.layer_type() << " in:" << to.in_size() << "(" << to.in_shape() << "), " <<
                                             "out:" << to.out_size() << "(" << to.out_shape() << ")" << std::endl;
    os << from.out_size() << " != " << to.in_size() << std::endl;
    std::string detail_info = os.str();

    throw nn_error("layer dimension mismatch!" + detail_info);
}

inline void data_mismatch(const layer_base& layer, const vec_t& data) {
    std::ostringstream os;

    os << std::endl;
    os << "data dimension:    " << data.size() << std::endl;
    os << "network dimension: " << layer.in_size() << "(" << layer.layer_type() << ":" << layer.in_shape() << ")" << std::endl;

    std::string detail_info = os.str();

    throw nn_error("input dimension mismath!" + detail_info);
}

inline void pooling_size_mismatch(layer_size_t in_width, layer_size_t in_height, layer_size_t pooling_size) {
    std::ostringstream os;

    os << std::endl;
    os << "WxH:" << in_width << "x" << in_height << std::endl;
    os << "pooling-size:" << pooling_size << std::endl;

    std::string detail_info = os.str();

    throw nn_error("width/height must be multiples of pooling size" + detail_info);
}

} // namespace tiny_cnn
