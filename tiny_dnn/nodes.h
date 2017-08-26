/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <algorithm>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef CNN_NO_SERIALIZATION
#include <cereal/types/tuple.hpp>
#include <cereal/types/utility.hpp>
#endif

#include "tiny_dnn/layers/layer.h"
#include "tiny_dnn/optimizers/optimizer.h"
#include "tiny_dnn/util/util.h"

namespace cereal {

template <typename Archive>
void save(Archive &ar, const std::vector<tiny_dnn::layer *> &v) {
#ifndef CNN_NO_SERIALIZATION
  ar(cereal::make_size_tag(static_cast<cereal::size_type>(v.size())));
  for (auto n : v) {
    tiny_dnn::layer::save_layer(ar, *n);
  }
#else
  throw tiny_dnn::nn_error("tiny-dnn was not built with Serialization support");
#endif  // CNN_NO_SERIALIZATION
}

template <typename Archive>
void load(Archive &ar, std::vector<std::shared_ptr<tiny_dnn::layer>> &v) {
#ifndef CNN_NO_SERIALIZATION
  cereal::size_type size;
  ar(cereal::make_size_tag(size));

  for (size_t i = 0; i < size; i++) {
    v.emplace_back(tiny_dnn::layer::load_layer(ar));
  }
#else
  throw tiny_dnn::nn_error("tiny-dnn was not built with Serialization support");
#endif  // CNN_NO_SERIALIZATION
}

}  // namespace cereal

namespace tiny_dnn {

/** basic class of various network types (sequential, multi-in/multi-out).
 *
 * this class holds list of pointer of Node, and provides entry point of
 * forward / backward operations.
 * Node is a computational unit of tiny-dnn (for example, convolution).
 * Currently 2 kinds of implementation are available: sequential and graph.
 *
 * Nodes can accept lvalue, rvalue and shared_ptr forms of node.
 * If given type is rvalue or shared_ptr, nodes create shared_ptr<node> to keep
 * given node alive. If given type is lvalue, tiny-dnn holds raw-pointer only
 * (to avoid double-free).
 *
 *     sequential s;
 *     s.add(fc(100, 200));                          // rvalue, moved into nodes
 *
 *     s.add(std::make_shared<fc>(200, 100));        // shared_ptr, shared by
 *nodes
 *
 *     fc out(100, 10);
 *     softmax sft();
 *     s.add(out);                                   // lvalue, hold raw-pointer
 *only
 *
 **/
class nodes {
 public:
  typedef std::vector<layer *>::iterator iterator;
  typedef std::vector<layer *>::const_iterator const_iterator;

  /**
   * propagate gradient
   * @param first        : gradient of cost function(dE/dy)
   **/
  virtual void backward(const std::vector<Tensor<>> &first) = 0;

  /**
   * @param first input  : data vectors
   **/
  virtual Tensor<> forward(const std::vector<Tensor<>> &first) = 0;  // NOLINT

  /**
   * update parameters and clear all gradients
   **/
  virtual void update_parameters(optimizer *opt, int batch_size) {
    for (auto l : nodes_) {
      l->update_parameters(opt, batch_size);
    }
  }

  /**
   * setup all weights, must be called before forward/backward
   **/
  virtual void setup(bool reset_weight) {
    for (auto l : nodes_) {
      l->setup(reset_weight);
    }
  }

  void clear_grads() {
    for (auto l : nodes_) {
      l->clear_grads();
    }
  }

  size_t size() const { return nodes_.size(); }
  iterator begin() { return nodes_.begin(); }
  iterator end() { return nodes_.end(); }
  const_iterator begin() const { return nodes_.begin(); }
  const_iterator end() const { return nodes_.end(); }
  layer *operator[](size_t index) { return nodes_[index]; }
  const layer *operator[](size_t index) const { return nodes_[index]; }
  size_t in_data_size() const { return nodes_.front()->in_data_size(); }
  size_t out_data_size() const { return nodes_.back()->out_data_size(); }

  template <typename T>
  const T &at(size_t index) const {
    const T *v = dynamic_cast<const T *>(nodes_[index]);
    if (v) return *v;
    throw nn_error("failed to cast");
  }

  template <typename T>
  T &at(size_t index) {
    T *v = dynamic_cast<T *>(nodes_[index]);
    if (v) return *v;
    throw nn_error("failed to cast");
  }

  // @todo: multiple output
  virtual float_t target_value_min(int out_channel = 0) const {
    CNN_UNREFERENCED_PARAMETER(out_channel);
    return nodes_.back()->out_value_range().first;
  }

  virtual float_t target_value_max(int out_channel = 0) const {
    CNN_UNREFERENCED_PARAMETER(out_channel);
    return nodes_.back()->out_value_range().second;
  }

  void save(std::ostream &os) const {  // NOLINT
    for (auto &l : nodes_) {
      l->save(os);
    }
  }

  void load(std::istream &is) {  // NOLINT
    setup(false);
    for (auto &l : nodes_) {
      l->load(is);
    }
  }

  /**
   * Converts vactor of labels to one-hot tensor
   * @param labels
   * @param ten
   */
  template <typename T = label_t, typename S>
  void label2vec(const Tensor<T> &labels, Tensor<float_t, S> &ten) const {
    // TODO(Randl): shape?
    assert(ten.shape()[0] == labels.size());
    assert(ten.shape()[1] == 1);
    assert(ten.shape()[2] == out_data_size());
    ten.fill(target_value_min());
    for (size_t i = 0; i < labels.size(); ++i) {
      ten.host_at(0, i, labels.host_at(i)) = target_value_max();
    }
  }

  template <typename OutputArchive>
  void save_model(OutputArchive &oa) const;

  template <typename InputArchive>
  void load_model(InputArchive &ia);

  template <typename OutputArchive>
  void save_weights(OutputArchive &oa) const {
    for (auto n : nodes_) {
      oa(*n);
    }
  }

  template <typename InputArchive>
  void load_weights(InputArchive &ia) {
    for (auto n : nodes_) {
      ia(*n);
    }
  }

 protected:
  template <typename T>
  void push_back(T &&node) {
    push_back_impl(
      std::forward<T>(node),
      typename std::is_rvalue_reference<decltype(node)>::type());  // NOLINT
  }

  template <typename T>
  void push_back(std::shared_ptr<T> node) {
    own_nodes_.push_back(node);
    nodes_.push_back(own_nodes_.back().get());
  }

  /**
   * transform indexing so that it's more suitable for per-layer operations
   * @param input [sample][channel][feature]
   * @param output [channel][sample][feature]
   */
  void reorder_for_layerwise_processing(const std::vector<Tensor<>> &input,
                                        std::vector<Tensor<>> &output) {
    size_t sample_count  = input.size();
    size_t channel_count = input[0].shape()[0];
    size_t input_size    = input[0].shape()[1];

    output.resize(channel_count);
    for (size_t i = 0; i < channel_count; ++i) {
      output[i].reshape({sample_count, input_size});
    }

    for (size_t sample = 0; sample < sample_count; ++sample) {
      assert(input[sample].shape()[0] == channel_count);
      for (size_t channel = 0; channel < channel_count; ++channel) {
        auto from =
          input[sample].subView(TensorSingleIndex(channel), TensorAll());
        auto to =
          output[channel].subView(TensorSingleIndex(sample), TensorAll());
        to.assign(from);
      }
    }
  }

  template <typename T>
  void push_back_impl(T &&node, std::true_type) {  // is_rvalue_reference
    own_nodes_.push_back(
      std::make_shared<typename std::remove_reference<T>::type>(
        std::forward<T>(node)));
    nodes_.push_back(own_nodes_.back().get());
  }

  template <typename T>
  void push_back_impl(T &&node, std::false_type) {
    nodes_.push_back(&node);
  }

  /* Nodes which this class has ownership */
  std::vector<std::shared_ptr<layer>> own_nodes_;
  /* List of all nodes which includes own_nodes */
  std::vector<layer *> nodes_;
};

/**
 * single-input, single-output feedforward network
 **/
class sequential : public nodes {
 public:
  void backward(const std::vector<Tensor<>> &first) override {
    std::vector<Tensor<>> reordered_grad;
    reorder_for_layerwise_processing(first, reordered_grad);
    assert(reordered_grad.size() == 1);

    nodes_.back()->set_out_grads(&reordered_grad[0], 1);

    for (auto l = nodes_.rbegin(); l != nodes_.rend(); l++) {
      (*l)->backward();
    }
  }

  Tensor<> forward(const std::vector<Tensor<>> &first) override {
    std::vector<Tensor<>> reordered_data;  // TODO(Randl): Tensor<>
    reorder_for_layerwise_processing(first, reordered_data);
    assert(reordered_data.size() == 1);
    nodes_.front()->set_in_data(&reordered_data[0], 1);

    for (auto l : nodes_) {
      l->forward();
    }

    std::vector<const Tensor<> *> out;
    nodes_.back()->output(out);

    return normalize_out(out);
  }

  template <typename T>
  void add(T &&layer) {
    push_back(std::forward<T>(layer));

    if (nodes_.size() != 1) {
      auto head = nodes_[nodes_.size() - 2];
      auto tail = nodes_[nodes_.size() - 1];
      connect(head, tail, 0, 0);
      auto out = head->outputs();
      auto in  = tail->inputs();
    }
    check_connectivity();
  }

  void check_connectivity() {
    for (size_t i = 0; i < nodes_.size() - 1; i++) {
      auto out = nodes_[i]->outputs();
      auto in  = nodes_[i + 1]->inputs();

      if (out[0] != in[0]) {
        throw nn_error("");
      }
    }
  }

  template <typename InputArchive>
  void load_connections(InputArchive &ia) {
    CNN_UNREFERENCED_PARAMETER(ia);
    for (size_t i = 0; i < nodes_.size() - 1; i++) {
      auto head = nodes_[i];
      auto tail = nodes_[i + 1];
      connect(head, tail, 0, 0);
    }
  }

  template <typename OutputArchive>
  void save_connections(OutputArchive &) const {}

 private:
  friend class nodes;

  /**
   * normalize indexing back to [sample][layer][feature]
   * @param out
   * @return
   */
  Tensor<> normalize_out(const std::vector<const Tensor<> *> &out) {
    const size_t sample_count = out[0]->shape()[0];
    const size_t feature_size = out[0]->shape()[1];
    Tensor<> normalized_output({sample_count, 1, feature_size});

    for (size_t sample = 0; sample < sample_count; ++sample) {
      auto from = out[0]->subView(TensorSingleIndex(sample), TensorAll());
      auto to   = normalized_output.subView(TensorSingleIndex(sample),
                                          TensorSingleIndex(0), TensorAll());
      to.assign(from);
    }

    return normalized_output;
  }
};

/**
 * generic graph network
 * @todo not implemented
 **/
class graph : public nodes {
 public:
  void backward(const std::vector<Tensor<>> &out_grad) override {
    size_t output_channel_count = out_grad[0].shape()[0];
    if (output_channel_count != output_layers_.size()) {
      throw nn_error("input size mismatch");
    }

    std::vector<Tensor<>> reordered_grad;
    reorder_for_layerwise_processing(out_grad, reordered_grad);
    assert(reordered_grad.size() == output_channel_count);

    for (size_t i = 0; i < output_channel_count; i++) {
      output_layers_[i]->set_out_grads(&reordered_grad[i], 1);
    }

    for (auto l = nodes_.rbegin(); l != nodes_.rend(); l++) {
      (*l)->backward();
    }
  }

  Tensor<> forward(const std::vector<Tensor<>> &in_data) override {
    size_t input_data_channel_count = in_data[0].shape()[0];

    if (input_data_channel_count != input_layers_.size()) {
      throw nn_error("input size mismatch");
    }

    std::vector<Tensor<>> reordered_data;
    reorder_for_layerwise_processing(in_data, reordered_data);
    assert(reordered_data.size() == input_data_channel_count);

    for (size_t channel_index = 0; channel_index < input_data_channel_count;
         channel_index++) {
      input_layers_[channel_index]->set_in_data(&reordered_data[channel_index],
                                                1);
    }

    for (auto l : nodes_) {
      l->forward();
    }
    return merge_outs();
  }

  void construct(const std::vector<layer *> &input,
                 const std::vector<layer *> &output) {
    std::vector<layer *> sorted;
    std::vector<node *> input_nodes(input.begin(), input.end());
    std::unordered_map<node *, std::vector<uint8_t>> removed_edge;

    // topological-sorting
    while (!input_nodes.empty()) {
      sorted.push_back(dynamic_cast<layer *>(input_nodes.back()));
      input_nodes.pop_back();

      layer *curr              = sorted.back();
      std::vector<node *> next = curr->next_nodes();

      for (size_t i = 0; i < next.size(); i++) {
        if (!next[i]) continue;
        // remove edge between next[i] and current
        if (removed_edge.find(next[i]) == removed_edge.end()) {
          removed_edge[next[i]] =
            std::vector<uint8_t>(next[i]->prev_nodes().size(), 0);
        }

        std::vector<uint8_t> &removed = removed_edge[next[i]];
        removed[find_index(next[i]->prev_nodes(), curr)] = 1;

        if (std::all_of(removed.begin(), removed.end(),
                        [](uint8_t x) { return x == 1; })) {
          input_nodes.push_back(next[i]);
        }
      }
    }

    for (auto &n : sorted) {
      nodes_.push_back(n);
    }

    input_layers_  = input;
    output_layers_ = output;

    setup(false);
  }

 private:
  friend class nodes;

  struct _graph_connection {
    void add_connection(size_t head,
                        size_t tail,
                        size_t head_index,
                        size_t tail_index) {
      if (!is_connected(head, tail, head_index, tail_index)) {
        connections.emplace_back(head, tail, head_index, tail_index);
      }
    }

    bool is_connected(size_t head,
                      size_t tail,
                      size_t head_index,
                      size_t tail_index) const {
      return std::find(connections.begin(), connections.end(),
                       std::make_tuple(head, tail, head_index, tail_index)) !=
             connections.end();
    }

    template <typename Archive>
    void serialize(Archive &ar) {
#ifndef CNN_NO_SERIALIZATION
      ar(CEREAL_NVP(connections), CEREAL_NVP(in_nodes), CEREAL_NVP(out_nodes));
#else
      throw nn_error("TinyDNN was not built with Serialization support");
#endif  // CNN_NO_SERIALIZATION
    }

    std::vector<std::tuple<size_t, size_t, size_t, size_t>> connections;
    std::vector<size_t> in_nodes, out_nodes;
  };

  template <typename OutputArchive>
  void save_connections(OutputArchive &oa) const {
#ifndef CNN_NO_SERIALIZATION
    _graph_connection gc;
    std::unordered_map<node *, size_t> node2id;
    size_t idx = 0;

    for (auto n : nodes_) {
      node2id[n] = idx++;
    }
    for (auto l : input_layers_) {
      gc.in_nodes.push_back(node2id[l]);
    }
    for (auto l : output_layers_) {
      gc.out_nodes.push_back(node2id[l]);
    }

    for (auto l : input_layers_) {
      graph_traverse(l, [=](layer &l) { CNN_UNREFERENCED_PARAMETER(l); },
                     [&](edge &e) {
                       auto next         = e.next();
                       size_t head_index = e.prev()->next_port(e);

                       for (auto n : next) {
                         size_t tail_index = n->prev_port(e);
                         gc.add_connection(node2id[e.prev()], node2id[n],
                                           head_index, tail_index);
                       }
                     });
    }

    oa(cereal::make_nvp("graph", gc));
#else
    throw nn_error("TinyDNN was not built with Serialization support");
#endif  // CNN_NO_SERIALIZATION
  }

  template <typename InputArchive>
  void load_connections(InputArchive &ia) {
#ifndef CNN_NO_SERIALIZATION
    _graph_connection gc;
    ia(cereal::make_nvp("graph", gc));

    for (auto c : gc.connections) {
      size_t head, tail, head_index, tail_index;
      std::tie(head, tail, head_index, tail_index) = c;
      connect(nodes_[head], nodes_[tail], head_index, tail_index);
    }
    for (auto in : gc.in_nodes) {
      input_layers_.push_back(nodes_[in]);
    }
    for (auto out : gc.out_nodes) {
      output_layers_.push_back(nodes_[out]);
    }
#else
    throw nn_error("TinyDNN was not built with Serialization support");
#endif  // CNN_NO_SERIALIZATION
  }

  /**
   * normalize indexing back to [sample][layer][feature]
   * @return
   */
  Tensor<> merge_outs() {
    Tensor<> merged;
    std::vector<const Tensor<> *> out;
    size_t output_channel_count = output_layers_.size();
    for (size_t output_channel = 0; output_channel < output_channel_count;
         ++output_channel) {
      output_layers_[output_channel]->output(out);

      const size_t sample_count = out[0]->shape()[0];
      const size_t feature_size = out[0]->shape()[1];
      if (output_channel == 0) {
        assert(merged.empty());
        merged.reshape({sample_count, output_channel_count, feature_size});
      }

      assert(merged.shape()[0] == sample_count);

      for (size_t sample = 0; sample < sample_count; ++sample) {
        auto to =
          merged.subView(TensorSingleIndex(sample),
                         TensorSingleIndex(output_channel), TensorAll());
        auto from = out[0]->subView(TensorSingleIndex(sample), TensorAll());
        to.assign(from);
      }
    }
    return merged;
  }

  size_t find_index(const std::vector<node *> &nodes, layer *target) {
    for (size_t i = 0; i < nodes.size(); i++) {
      if (nodes[i] == static_cast<node *>(&*target)) return i;
    }
    throw nn_error("invalid connection");
  }
  std::vector<layer *> input_layers_;
  std::vector<layer *> output_layers_;
};

template <typename OutputArchive>
void nodes::save_model(OutputArchive &oa) const {
#ifndef CNN_NO_SERIALIZATION
  oa(cereal::make_nvp("nodes", nodes_));

  if (typeid(*this) == typeid(sequential)) {
    dynamic_cast<const sequential *>(this)->save_connections(oa);
  } else {
    dynamic_cast<const graph *>(this)->save_connections(oa);
  }
#else
  throw nn_error("TinyDNN was not built with Serialization support");
#endif  // CNN_NO_SERIALIZATION
}

template <typename InputArchive>
void nodes::load_model(InputArchive &ia) {
#ifndef CNN_NO_SERIALIZATION
  own_nodes_.clear();
  nodes_.clear();

  ia(cereal::make_nvp("nodes", own_nodes_));

  for (auto &n : own_nodes_) {
    nodes_.push_back(&*n);
  }

  if (typeid(*this) == typeid(sequential)) {
    dynamic_cast<sequential *>(this)->load_connections(ia);
  } else {
    dynamic_cast<graph *>(this)->load_connections(ia);
  }
#else
  throw nn_error("TinyDNN was not built with Serialization support");
#endif  // CNN_NO_SERIALIZATION
}

}  // namespace tiny_dnn
