/*
    Copyright (c) 2016, Taiga Nomi
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
#include <memory>
#include <numeric>
#include <vector>
#include <set>
#include <queue>
#include <unordered_set>

#include "tiny_cnn/util/util.h"
#include "tiny_cnn/util/product.h"
#include "tiny_cnn/util/image.h"
#include "tiny_cnn/util/weight_init.h"
#include "tiny_cnn/optimizers/optimizer.h"

#include "tiny_cnn/activations/activation_function.h"

namespace tiny_cnn {

class node;
class layer;
class edge;

typedef node* nodeptr_t;
typedef std::shared_ptr<edge> edgeptr_t;

typedef layer* layerptr_t;

/**
 * base class of all kind of tinny-cnn data
 **/
class node : public std::enable_shared_from_this<node> {
public:
    node(cnn_size_t in_size, cnn_size_t out_size)
        : prev_(in_size), next_(out_size) {}
    virtual ~node() {}

    const std::vector<edgeptr_t>& prev() const { return prev_; }
    const std::vector<edgeptr_t>& next() const { return next_; }

    cnn_size_t prev_port(const edge& e) const {
        auto it = std::find_if(prev_.begin(), prev_.end(),
                               [&](edgeptr_t ep) { return ep.get() == &e; });
        return (cnn_size_t)std::distance(prev_.begin(), it);
    }

    cnn_size_t next_port(const edge& e) const {
        auto it = std::find_if(next_.begin(), next_.end(),
                               [&](edgeptr_t ep) { return ep.get() == &e; });
        return (cnn_size_t)std::distance(next_.begin(), it);
    }

    std::vector<node*> prev_nodes() const; // @todo refactor and remove this method
    std::vector<node*> next_nodes() const; // @todo refactor and remove this method
 protected:
    node() = delete;

    friend void connect(layerptr_t head, layerptr_t tail,
                        cnn_size_t head_index, cnn_size_t tail_index);

    mutable std::vector<edgeptr_t> prev_;
    mutable std::vector<edgeptr_t> next_;
};

/**
 * class containing input/output data
 **/
class edge {
 public:
    edge(node* prev, const shape3d& shape, vector_type vtype)
        : worker_specific_data_(!is_trainable_weight(vtype)),
          shape_(shape),
          vtype_(vtype),
          data_(1, vec_t(shape.size())),
          prev_(prev) {
      grad_.resize(1, vec_t(shape.size()));
    }

    void merge_grads(cnn_size_t worker_size, vec_t *dst) {
        *dst = grad_[0];

        for (cnn_size_t i = 1; i < worker_size; i++) {
            vectorize::reduce<float_t>(&grad_[i][0], dst->size(), &(*dst)[0]);
        }
    }

    void clear_grads(cnn_size_t worker_size) {
        for (cnn_size_t i = 0; i < worker_size; i++)
            clear_grad_onwork(i);
    }

    void clear_grad_onwork(cnn_size_t index) {
        std::fill(grad_[index].begin(), grad_[index].end(), (float_t)0);
    }

    void set_worker_size(cnn_size_t size) {
        if (worker_specific_data_) data_.resize(size, data_[0]);
        grad_.resize(size, grad_[0]);
    }

    vec_t* get_data(cnn_size_t worker_index = 0) {
        return worker_specific_data_ ? &data_[worker_index] : &data_[0];
    }

    const vec_t* get_data(cnn_size_t worker_index = 0) const {
        return worker_specific_data_ ? &data_[worker_index] : &data_[0];
    }

    vec_t* get_gradient(cnn_size_t worker_index = 0) {
        return &grad_[worker_index];
    }

    const vec_t* get_gradient(cnn_size_t worker_index = 0) const {
        return &grad_[worker_index];
    }

    const std::vector<node*>& next() const { return next_; }
    node* prev() { return prev_; }
    const node* prev() const { return prev_; }

    const shape3d& shape() const { return shape_; }
    vector_type vtype() const { return vtype_; }
    void add_next_node(node* next) { next_.push_back(next); }

 private:
    bool worker_specific_data_;
    shape3d shape_;
    vector_type vtype_;
    std::vector<vec_t> data_;
    std::vector<vec_t> grad_;
    node* prev_;               // previous node, "producer" of this tensor
    std::vector<node*> next_;  // next nodes, "consumers" of this tensor
};

inline std::vector<node*> node::prev_nodes() const {
    std::set<node*> sets;
    for (auto& e : prev_) {
        if (e && e->prev()) sets.insert(e->prev());
    }
    return std::vector<node*>(sets.begin(), sets.end());
}

inline std::vector<node*> node::next_nodes() const {
    std::set<node*> sets;
    for (auto& e : next_) {
        if (e) {
            auto n = e->next();
            sets.insert(n.begin(), n.end());
        }
    }
    return std::vector<node*>(sets.begin(), sets.end());
}

template <typename T>
struct node_tuple {
    node_tuple(T l1, T l2) {
        nodes_.push_back(l1); nodes_.push_back(l2);
    }
    std::vector<T> nodes_;
};

template <typename T>
node_tuple<T*> operator , (T& l1, T& l2) {
    return node_tuple<T*>(&l1, &l2);
}

template <typename T>
node_tuple<std::shared_ptr<T>> operator , (std::shared_ptr<T> l1, std::shared_ptr<T> l2) {
    return node_tuple<std::shared_ptr<T>>(l1, l2);
}

template <typename T>
node_tuple<std::shared_ptr<T>> operator , (node_tuple<std::shared_ptr<T>> lhs, std::shared_ptr<T>& rhs) {
    lhs.nodes_.push_back(rhs);
    return lhs;
}

template <typename T>
node_tuple<T*> operator , (node_tuple<T*> lhs, T& rhs) {
    lhs.nodes_.push_back(&rhs);
    return lhs;
}

template <typename T, typename U>
inline std::shared_ptr<U>& operator << (std::shared_ptr<T>& lhs,
                                        std::shared_ptr<U>& rhs) {
    connect(lhs.get(), rhs.get());
    return rhs;
}

template <typename T, typename U>
inline U& operator << (const node_tuple<T>& lhs, U& rhs) {
    for (size_t i = 0; i < lhs.nodes_.size(); i++) {
        connect(&*lhs.nodes_[i], &*rhs, 0, i);
    }
    return rhs;
}

template <typename T, typename U>
inline node_tuple<T>& operator << (U& lhs, const node_tuple<T>& rhs) {
    for (size_t i = 0; i < rhs.nodes_.size(); i++) {
        connect(&*lhs, &*rhs.nodes_[i], i, 0);
    }
    return rhs;
}

template <typename T, typename U>
inline U& operator << (const node_tuple<T*>& lhs, U& rhs) {
    for (size_t i = 0; i < lhs.nodes_.size(); i++) {
        connect(lhs.nodes_[i], &rhs, 0, i);
    }
    return rhs;
}

template <typename T, typename U>
inline node_tuple<T*>& operator << (U& lhs, const node_tuple<T*>& rhs) {
    for (size_t i = 0; i < rhs.nodes_.size(); i++) {
        connect(&lhs, rhs.nodes_[i], i, 0);
    }
    return rhs;
}


}   // namespace tiny_cnn
