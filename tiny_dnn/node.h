/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <algorithm>
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

#ifdef DNN_USE_IMAGE_API
#include "tiny_dnn/util/image.h"
#endif

namespace tiny_dnn {

class node;
class layer;
class edge;

typedef std::shared_ptr<edge> edgeptr_t;

/**
 * base class of all kind of tinny-cnn data
 **/
class node : public std::enable_shared_from_this<node> {
 public:
  node(size_t in_size, size_t out_size) : prev_(in_size), next_(out_size) {}
  virtual ~node() {}

  const std::vector<edgeptr_t> &prev() const { return prev_; }
  const std::vector<edgeptr_t> &next() const { return next_; }

  size_t prev_port(const edge &e) const {
    auto it = std::find_if(prev_.begin(), prev_.end(),
                           [&](edgeptr_t ep) { return ep.get() == &e; });
    return (size_t)std::distance(prev_.begin(), it);
  }

  size_t next_port(const edge &e) const {
    auto it = std::find_if(next_.begin(), next_.end(),
                           [&](edgeptr_t ep) { return ep.get() == &e; });
    return (size_t)std::distance(next_.begin(), it);
  }

  std::vector<node *> prev_nodes()
    const;  // @todo refactor and remove this method
  std::vector<node *> next_nodes()
    const;  // @todo refactor and remove this method

 protected:
  node() = delete;

  friend void connect(layer *head,
                      layer *tail,
                      size_t head_index,
                      size_t tail_index);

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
  std::vector<node *> vecs;
  for (auto &e : prev_) {
    if (e && e->prev()) {
      vecs.insert(vecs.end(), e->prev());
    }
  }
  return vecs;
}

inline std::vector<node *> node::next_nodes() const {
  std::vector<node *> vecs;
  for (auto &e : next_) {
    if (e) {
      auto n = e->next();
      vecs.insert(vecs.end(), n.begin(), n.end());
    }
  }
  return vecs;
}

template <typename T>
struct layer_tuple {
  layer_tuple(T l1, T l2) {
    layers_.push_back(l1);
    layers_.push_back(l2);
  }
  std::vector<T> layers_;
};

template <
  typename T,
  typename U,
  typename std::enable_if<std::is_base_of<layer, T>::value &&
                          std::is_base_of<layer, U>::value>::type * = nullptr>
layer_tuple<layer *> operator,(T &l1, U &l2) {
  return layer_tuple<layer *>(&l1, &l2);
}

template <
  typename T,
  typename U,
  typename std::enable_if<std::is_base_of<layer, T>::value &&
                          std::is_base_of<layer, U>::value>::type * = nullptr>
layer_tuple<std::shared_ptr<layer>> operator,(std::shared_ptr<T> l1,
                                              std::shared_ptr<U> l2) {
  return layer_tuple<std::shared_ptr<layer>>(l1, l2);
}

template <
  typename T,
  typename std::enable_if<std::is_base_of<layer, T>::value>::type * = nullptr>
layer_tuple<layer *> operator,(layer_tuple<layer *> lhs, T &rhs) {
  lhs.layers_.push_back(&rhs);
  return lhs;
}

template <
  typename T,
  typename std::enable_if<std::is_base_of<layer, T>::value>::type * = nullptr>
layer_tuple<std::shared_ptr<layer>> operator,(
  layer_tuple<std::shared_ptr<layer>> lhs, std::shared_ptr<T> &rhs) {
  lhs.layers_.push_back(rhs);
  return lhs;
}

template <
  typename T,
  typename std::enable_if<std::is_base_of<layer, T>::value>::type * = nullptr>
layer_tuple<layer *> operator,(T &lhs, layer_tuple<layer *> rhs) {
  rhs.layers_.insert(rhs.layers_.begin(), &lhs);
  return rhs;
}

template <
  typename T,
  typename std::enable_if<std::is_base_of<layer, T>::value>::type * = nullptr>
layer_tuple<std::shared_ptr<layer>> operator,(
  std::shared_ptr<T> &lhs, layer_tuple<std::shared_ptr<layer>> rhs) {
  rhs.layers_.insert(rhs.layers_.begin(), lhs);
  return rhs;
}

template <typename T, typename U>
inline std::shared_ptr<U> &operator<<(std::shared_ptr<T> &lhs,
                                      std::shared_ptr<U> &rhs) {
  connect(lhs.get(), rhs.get());
  return rhs;
}

template <typename T>
inline T &operator<<(const layer_tuple<std::shared_ptr<layer>> &lhs, T &rhs) {
  for (size_t i = 0; i < lhs.layers_.size(); i++) {
    connect(&*lhs.layers_[i], &*rhs, 0, i);
  }
  return rhs;
}

template <typename T>
inline const layer_tuple<std::shared_ptr<layer>> &operator<<(
  T &lhs, const layer_tuple<std::shared_ptr<layer>> &rhs) {
  for (size_t i = 0; i < rhs.layers_.size(); i++) {
    connect(&*lhs, &*rhs.layers_[i], i, 0);
  }
  return rhs;
}

template <typename T>
inline T &operator<<(const layer_tuple<layer *> &lhs, T &rhs) {
  for (size_t i = 0; i < lhs.layers_.size(); i++) {
    connect(lhs.layers_[i], &rhs, 0, i);
  }
  return rhs;
}

template <typename T>
inline const layer_tuple<layer *> &operator<<(T &lhs,
                                              const layer_tuple<layer *> &rhs) {
  for (size_t i = 0; i < rhs.layers_.size(); i++) {
    connect(&lhs, rhs.layers_[i], i, 0);
  }
  return rhs;
}
}  // namespace tiny_dnn
