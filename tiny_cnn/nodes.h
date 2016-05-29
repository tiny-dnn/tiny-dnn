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

#include <vector>
#include <unordered_map>

#include "tiny_cnn/util/util.h"
#include "tiny_cnn/layers/layer.h"
#include "tiny_cnn/optimizers/optimizer.h"

namespace tiny_cnn {

/** basic class of various network types (sequential, multi-in/multi-out).
 *
 * this class holds list of pointer of Node, and provides entry point of
 * forward / backward operations.
 * Node is a computational unit of tiny-cnn (for example, convolution).
 * Currently 2 kinds of implementation are available: sequential and graph.
 *
 * Nodes can accept lvalue, rvalue and shared_ptr forms of node.
 * If given type is rvalue or shared_ptr, nodes create shared_ptr<node> to keep
 * given node alive. If given type is lvalue, tiny-cnn holds raw-pointer only
 * (to avoid double-free).
 *
 *     sequential s;
 *     s.add(fc<tan_h>(100, 200));                   // rvalue, moved into nodes
 *
 *     s.add(std::make_shared<fc<tan_h>>(200, 100)); // shared_ptr, shared by nodes
 *
 *     fc<softmax> out(100, 10);
 *     s.add(out);                                   // lvalue, hold raw-pointer only
 *
 **/
class nodes {
 public:
     typedef typename std::vector<layerptr_t>::iterator iterator;
     typedef typename std::vector<layerptr_t>::const_iterator const_iterator;

    /**
     * propagate gradient
     * @param first        : gradient of cost function(dE/dy)
     * @param worker_index : id of worker-task
     **/
    virtual
    void backward(const std::vector<vec_t>& first, int worker_index) = 0;

    /**
     * @param first input  : data vectors
     * @param worker_index : id of worker-task
     **/
    virtual
    std::vector<vec_t> forward(const std::vector<vec_t>& first, int worker_index) = 0; // NOLINT

    /**
     * update weights and clear all gradients
     **/
    virtual
    void update_weights(optimizer *opt, int num_workers, int batch_size) {
        for (auto l : nodes_) {
            l->update_weight(opt, num_workers, batch_size);
        }
    }

    /**
     * change max number of task
     **/
    virtual void set_worker_count(cnn_size_t worker) {
        for (auto l : nodes_) {
            l->set_worker_count(worker);
        }
    }

    /**
     * setup all weights, must be called before forward/backward
     **/
    virtual void setup(bool reset_weight, int max_task_size) {
        for (auto l : nodes_) {
            l->setup(reset_weight, max_task_size);
        }
    }

    void clear_grads(int max_task_size) {
        for (auto l : nodes_) {
            l->clear_grads(max_task_size);
        }
    }

    size_t size() const { return nodes_.size(); }
    iterator begin() { return nodes_.begin(); }
    iterator end() { return nodes_.end(); }
    const_iterator begin() const { return nodes_.begin(); }
    const_iterator end() const { return nodes_.end(); }
    layer* operator[] (size_t index) { return nodes_[index]; }
    const layer* operator[] (size_t index) const { return nodes_[index]; }
    cnn_size_t in_data_size() const { return nodes_.front()->in_data_size(); }
    cnn_size_t out_data_size() const { return nodes_.back()->out_data_size(); }

    template <typename T>
    const T& at(size_t index) const {
        const T* v = dynamic_cast<const T*>(nodes_[index]);
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

    virtual void save(std::ostream& os) const { // NOLINT
        for (auto& l : nodes_) {
            l->save(os);
        }
    }

    virtual void load(std::istream& is) { // NOLINT
        setup(false, 1);
        for (auto& l : nodes_) {
            l->load(is);
        }
    }

    virtual void load(const std::vector<float_t>& vec) {
        int idx = 0;
        setup(false, 1);
        for (auto& l : nodes_) {
            l->load(vec, idx);
        }
    }

    void label2vec(const label_t* t, int num, std::vector<vec_t> *vec) const {
        cnn_size_t outdim = out_data_size();

        vec->reserve(num);
        for (int i = 0; i < num; i++) {
            assert(t[i] < outdim);
            vec->emplace_back(outdim, target_value_min());
            vec->back()[t[i]] = target_value_max();
        }
    }

 protected:
    template <typename T>
    void push_back(T&& node) {
        push_back_impl(std::forward<T>(node),
                       typename std::is_rvalue_reference<decltype(node)>::type()); // NOLINT
    }

    template <typename T>
    void push_back(std::shared_ptr<T> node) {
        own_nodes_.push_back(node);
        nodes_.push_back(own_nodes_.back().get());
    }

 protected:
    template <typename T>
    void push_back_impl(T&& node, std::true_type) {  // is_rvalue_reference
        own_nodes_.push_back(std::make_shared<
            typename std::remove_reference<T>::type>(std::forward<T>(node)));
        nodes_.push_back(own_nodes_.back().get());
    }

    template <typename T>
    void push_back_impl(T&& node, std::false_type) {
        nodes_.push_back(&node);
    }

    /* Nodes which this class has ownership */
    std::vector<std::shared_ptr<layer>> own_nodes_;
    /* List of all nodes which includes own_nodes */
    std::vector<layerptr_t> nodes_;
};

/**
 * single-input, single-output feedforward network
 **/
class sequential : public nodes {
 public:
    void backward(const std::vector<vec_t>& first, int worker_index) override {
        nodes_.back()->set_out_grads(&first[0], first.size(), worker_index);

        for (auto l = nodes_.rbegin(); l != nodes_.rend(); l++) {
            (*l)->backward(worker_index);
        }
    }

    std::vector<vec_t> forward(const std::vector<vec_t>& first,
                               int worker_index) override {
        nodes_.front()->set_in_data(&first[0], first.size(), worker_index);

        for (auto l : nodes_) {
            l->forward(worker_index);
        }

        return nodes_.back()->output(worker_index);
    }

    template <typename T>
    void add(T&& layer) {
        push_back(std::forward<T>(layer));

        if (nodes_.size() != 1) {
            auto head = nodes_[nodes_.size()-2];
            auto tail = nodes_[nodes_.size()-1];
            connect(head, tail, 0, 0);
            auto out = head->get_outputs();
            auto in = tail->get_inputs();
        }
        check_connectivity();
    }

    void check_connectivity() {
        for (cnn_size_t i = 0; i < nodes_.size() - 1; i++) {
            auto out = nodes_[i]->get_outputs();
            auto in = nodes_[i+1]->get_inputs();

            if (out[0] != in[0]) {
                throw nn_error("");
            }
        }
    }

 private:
};

/**
 * generic graph network
 * @todo not implemented
 **/
class graph : public nodes {
 public:
    void backward(const std::vector<vec_t>& out_grad,
                  int worker_index) override {
        if (out_grad.size() != output_layers_.size()) {
            throw nn_error("input size mismatch");
        }

        for (cnn_size_t i = 0; i < out_grad.size(); i++) {
            output_layers_[i]->set_out_grads(&out_grad[i], 1, worker_index);
        }

        for (auto l = nodes_.rbegin(); l != nodes_.rend(); l++) {
            (*l)->backward(worker_index);
        }
    }

    std::vector<vec_t> forward(const std::vector<vec_t>& in_data,
                               int worker_index) {
        if (in_data.size() != input_layers_.size()) {
            throw nn_error("input size mismatch");
        }

        for (cnn_size_t i = 0; i < in_data.size(); i++) {
            input_layers_[i]->set_in_data(&in_data[i], 1, worker_index);
        }

        for (auto l : nodes_) {
            l->forward(worker_index);
        }
        return merge_outs(worker_index);
    }

    void construct(const std::vector<layerptr_t>& input,
                   const std::vector<layerptr_t>& output) {
        std::vector<layerptr_t> sorted;
        std::vector<nodeptr_t> input_nodes(input.begin(), input.end());
        std::unordered_map<node*, std::vector<uint8_t>> removed_edge;

        // topological-sorting
        while (!input_nodes.empty()) {
            sorted.push_back(dynamic_cast<layerptr_t>(input_nodes.back()));
            input_nodes.pop_back();

            layerptr_t curr = sorted.back();
            std::vector<node*> next = curr->next_nodes();

            for (size_t i = 0; i < next.size(); i++) {
                if (!next[i]) continue;
                // remove edge between next[i] and current
                if (removed_edge.find(next[i]) == removed_edge.end()) {
                    removed_edge[next[i]] =
                        std::vector<uint8_t>(next[i]->prev_nodes().size(), 0);
                }

                std::vector<uint8_t>& removed = removed_edge[next[i]];
                removed[find_index(next[i]->prev_nodes(), curr)] = 1;

                if (std::all_of(removed.begin(), removed.end(), [](uint8_t x) {
                        return x == 1; })) {
                    input_nodes.push_back(next[i]);
                }
            }
        }

        for (auto& n : sorted) {
            nodes_.push_back(n);
        }

        input_layers_ = input;
        output_layers_ = output;

        setup(false, 1);
    }

 private:

     std::vector<vec_t> merge_outs(int worker_index) {
         std::vector<vec_t> merged;
         for (auto& l : output_layers_) {
             std::vector<vec_t> out = l->output(worker_index);
             merged.insert(merged.end(), out.begin(), out.end());
         }
         return merged;
     }

    cnn_size_t find_index(const std::vector<node*>& nodes,
                          layerptr_t target) {
        for (cnn_size_t i = 0; i < nodes.size(); i++) {
            if (nodes[i] == static_cast<node*>(&*target)) return i;
        }
        throw nn_error("invalid connection");
    }
    std::vector<layerptr_t> input_layers_;
    std::vector<layerptr_t> output_layers_;
};

}  // namespace tiny_cnn
