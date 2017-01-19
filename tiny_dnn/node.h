/*
    Copyright (c) 2016, Taiga Nomi
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <iomanip>
#include <memory>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <unordered_set>
#include <vector>

#include "tiny_dnn/optimizers/optimizer.h"
#include "tiny_dnn/util/product.h"
#include "tiny_dnn/util/util.h"
#include "tiny_dnn/util/weight_init.h"

#include "tiny_dnn/activations/activation_function.h"

#ifdef DNN_USE_IMAGE_API
#include "tiny_dnn/util/image.h"
#endif

namespace tiny_dnn {

class node;
class layer;
class edge;

typedef node *nodeptr_t;
typedef std::shared_ptr<edge> edgeptr_t;

typedef layer *layerptr_t;

/**
 * base class of all kind of tinny-cnn data
 **/
class node : public std::enable_shared_from_this<node> {
 public:
  node(serial_size_t in_size, serial_size_t out_size)
    : prev_(in_size), next_(out_size) {}
  virtual ~node() {}

  const std::vector<edgeptr_t> &prev() const { return prev_; }
  const std::vector<edgeptr_t> &next() const { return next_; }

  serial_size_t prev_port(const edge &e) const {
    auto it = std::find_if(prev_.begin(), prev_.end(),
                           [&](edgeptr_t ep) { return ep.get() == &e; });
    return (serial_size_t)std::distance(prev_.begin(), it);
  }

  serial_size_t next_port(const edge &e) const {
    auto it = std::find_if(next_.begin(), next_.end(),
                           [&](edgeptr_t ep) { return ep.get() == &e; });
    return (serial_size_t)std::distance(next_.begin(), it);
  }

  std::vector<node *> prev_nodes()
    const;  // @todo refactor and remove this method
  std::vector<node *> next_nodes()
    const;  // @todo refactor and remove this method

 protected:
  node() = delete;

  friend void connect(layerptr_t head,
                      layerptr_t tail,
                      serial_size_t head_index,
                      serial_size_t tail_index);

  mutable std::vector<edgeptr_t> prev_;
  mutable std::vector<edgeptr_t> next_;
};

/**
 * class containing input/output data
 **/
class edge {
 public:
  edge(node *prev, const shape3d &shape, vector_type vtype)
    : shape_(shape),
      vtype_(vtype),
      data_({vec_t(shape.size())}),
      grad_({vec_t(shape.size())}),
      prev_(prev) {}

  void merge_grads(vec_t *dst) {
    assert(!grad_.empty());
    const auto &grad_head = grad_[0];
    size_t sz             = grad_head.size();
    dst->resize(sz);
    float_t *pdst = &(*dst)[0];
    // dst = grad_[0]
    std::copy(grad_head.begin(), grad_head.end(), pdst);
    // @todo consider adding parallelism
    for (size_t sample = 1, sample_count = grad_.size(); sample < sample_count;
         ++sample) {
      // dst += grad_[sample]
      vectorize::reduce<float_t>(&grad_[sample][0], sz, pdst);
    }
  }

  void clear_grads() {
    for (size_t sample = 0, sample_count = grad_.size(); sample < sample_count;
         ++sample) {
      auto &g = grad_[sample];
      vectorize::fill(&g[0], g.size(), float_t{0});
    }
  }

  tensor_t *get_data() { return &data_; }

  const tensor_t *get_data() const { return &data_; }

  tensor_t *get_gradient() { return &grad_; }

  const tensor_t *get_gradient() const { return &grad_; }

  const std::vector<node *> &next() const { return next_; }
  node *prev() { return prev_; }
  const node *prev() const { return prev_; }

  const shape3d &shape() const { return shape_; }
  vector_type vtype() const { return vtype_; }
  void add_next_node(node *next) { next_.push_back(next); }

 private:
  shape3d shape_;
  vector_type vtype_;
  tensor_t data_;
  tensor_t grad_;
  node *prev_;                // previous node, "producer" of this tensor
  std::vector<node *> next_;  // next nodes, "consumers" of this tensor
};

inline std::vector<node *> node::prev_nodes() const {
  std::set<node *> sets;
  for (auto &e : prev_) {
    if (e && e->prev()) sets.insert(e->prev());
  }
  return std::vector<node *>(sets.begin(), sets.end());
}

inline std::vector<node *> node::next_nodes() const {
  std::set<node *> sets;
  for (auto &e : next_) {
    if (e) {
      auto n = e->next();
      sets.insert(n.begin(), n.end());
    }
  }
  return std::vector<node *>(sets.begin(), sets.end());
}

template <typename T>
struct node_tuple {
  node_tuple(T l1, T l2) {
    nodes_.push_back(l1);
    nodes_.push_back(l2);
  }
  std::vector<T> nodes_;
};

template <typename T>
node_tuple<T *> operator,(T &l1, T &l2) {
  return node_tuple<T *>(&l1, &l2);
}

template <typename T>
node_tuple<std::shared_ptr<T>> operator,(std::shared_ptr<T> l1,
                                         std::shared_ptr<T> l2) {
  return node_tuple<std::shared_ptr<T>>(l1, l2);
}

template <typename T>
node_tuple<std::shared_ptr<T>> operator,(node_tuple<std::shared_ptr<T>> lhs,
                                         std::shared_ptr<T> &rhs) {
  lhs.nodes_.push_back(rhs);
  return lhs;
}

template <typename T>
node_tuple<T *> operator,(node_tuple<T *> lhs, T &rhs) {
  lhs.nodes_.push_back(&rhs);
  return lhs;
}

template <typename T, typename U>
inline std::shared_ptr<U> &operator<<(std::shared_ptr<T> &lhs,
                                      std::shared_ptr<U> &rhs) {
  connect(lhs.get(), rhs.get());
  return rhs;
}

template <typename T, typename U>
inline U &operator<<(const node_tuple<T> &lhs, U &rhs) {
  for (serial_size_t i = 0; i < static_cast<serial_size_t>(lhs.nodes_.size());
       i++) {
    connect(&*lhs.nodes_[i], &*rhs, 0, i);
  }
  return rhs;
}

template <typename T, typename U>
inline node_tuple<T> &operator<<(U &lhs, const node_tuple<T> &rhs) {
  for (serial_size_t i = 0; i < static_cast<serial_size_t>(rhs.nodes_.size());
       i++) {
    connect(&*lhs, &*rhs.nodes_[i], i, 0);
  }
  return rhs;
}

template <typename T, typename U>
inline U &operator<<(const node_tuple<T *> &lhs, U &rhs) {
  for (serial_size_t i = 0; i < static_cast<serial_size_t>(lhs.nodes_.size());
       i++) {
    connect(lhs.nodes_[i], &rhs, 0, i);
  }
  return rhs;
}

template <typename T, typename U>
inline node_tuple<T *> &operator<<(U &lhs, const node_tuple<T *> &rhs) {
  for (serial_size_t i = 0; i < static_cast<serial_size_t>(rhs.nodes_.size());
       i++) {
    connect(&lhs, rhs.nodes_[i], i, 0);
  }
  return rhs;
}

}  // namespace tiny_dnn
