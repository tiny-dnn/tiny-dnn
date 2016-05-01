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

#include "tiny_cnn/util/util.h"
#include "tiny_cnn/layers/layer.h"
#include "tiny_cnn/optimizers/optimizer.h"
#include <unordered_map>

namespace tiny_cnn {

/**
 * basic class of various network types (sequential, multi-in/multi-out)
 **/
class nodes {
public:
    typedef std::shared_ptr<layer_base> nodeptr_t;
    typedef typename std::vector<nodeptr_t>::iterator iterator;
    typedef typename std::vector<nodeptr_t>::const_iterator const_iterator;

    /**
     * propagate gradient
     * @param first        : gradient of cost function(dE/dy)
     * @param worker_index : id of worker-task
     **/
    virtual void backward(const std::vector<vec_t>& first, int worker_index) = 0;

    /**
     * @param first input  : data vectors
     * @param worker_index : id of worker-task
     **/
    virtual std::vector<vec_t> forward(const std::vector<vec_t>& first, int worker_index) = 0;

    /**
     * update weights and clear all gradients
     **/
    virtual void update_weights(optimizer *opt, int num_workers, int batch_size) = 0;

    /**
     * change max number of task
     **/
    virtual void set_worker_count(cnn_size_t worker) = 0;

    /**
     * setup all weights, must be called before forward/backward
     **/
    virtual void setup(bool reset_weight, int max_task_size) = 0;

    iterator begin() { return nodes_.begin(); }
    iterator end() { return nodes_.end(); }
    const_iterator begin() const { return nodes_.begin(); }
    const_iterator end() const { return nodes_.end(); }

    cnn_size_t in_data_size() const { return nodes_.front()->in_data_size(); }
    cnn_size_t out_data_size() const { return nodes_.back()->out_data_size(); }

    // @todo: multiple output
    virtual float_t target_value_min(int out_channel = 0) const {
        return nodes_.back()->out_value_range().first;
    }
    virtual float_t target_value_max(int out_channel = 0) const {
        return nodes_.back()->out_value_range().second;
    }

    virtual void save(std::ostream& os) const = 0;
    virtual void load(std::istream& is) = 0;
    virtual void load(const std::vector<float_t>& vec) = 0;
protected:
    std::vector<nodeptr_t> nodes_;
};

/**
 * single-input, single-output feedforward network
 **/
class sequential : public nodes {
public:
    void backward(const std::vector<vec_t>& first, int worker_index) override {
        auto out_grad_id = nodes_.back()->out_grad_index();
        for (size_t i = 0; i < out_grad_id.size(); i++) {
            *storage_.get(out_grad_id[i], worker_index) = first[i];
        }

        for (auto l = nodes_.rbegin(); l != nodes_.rend(); l++) {
            (*l)->backward(&storage_, worker_index);
        }
    }

    std::vector<vec_t> forward(const std::vector<vec_t>& first, int worker_index) override {
        auto input_data_id = nodes_.front()->in_data_index();
        for (size_t i = 0; i < input_data_id.size(); i++) {
            *storage_.get(input_data_id[i], worker_index) = first[i];
        }

        for (auto l : nodes_) {
            l->forward(&storage_, worker_index);
        }

        return nodes_.back()->output(storage_, worker_index);
    }

    void update_weights(optimizer *opt, int num_workers, int batch_size) override {
        for (auto l : nodes_) {
            l->update_weight(opt, &storage_, num_workers, batch_size);
        }
    }

    void set_worker_count(cnn_size_t worker) {
        storage_.set_worker_size(worker);
        for (auto l : nodes_)
            l->set_worker_count(worker);
    }

    void setup(bool reset_weight, int max_task_size) {
        for (auto l : nodes_) {
            l->setup(&storage_, reset_weight, max_task_size);
        }
    }

    void add(nodeptr_t layer) {
        if (!nodes_.empty()) {
            nodes_.back()->connect(layer.get(), 0, 0, &storage_);
        }
        nodes_.push_back(layer);
    }

    void save(std::ostream& os) const {
        for (auto& l : nodes_)
            l->save(os, storage_);
    }

    void load(std::istream& is) {
        for (auto& l : nodes_)
            l->load(is, storage_);
    }

    void load(const std::vector<float_t>& vec) {
        int idx = 0;
        for (auto& l : nodes_)
            l->load(vec, idx, storage_);
    }
private:

     data_storage storage_;
};

/**
 * generic graph network
 * @todo not implemented
 **/
class graph : public nodes {
public:

    void backward(const std::vector<vec_t>& out_grad, int worker_index) override {

        // set input-data to first layers
        for (size_t i = 0; i < out_grad.size(); i++) {
            for (size_t j = 0; j < output2grads_[i].size(); j++) {
                vec_t *v = storage_.get(output2grads_[i][j], worker_index);
                *v = out_grad[i];
            }
        }

        for (int i = (int)layer_names_.size() - 1; i >= 0; i--)
            get_by_bame(layer_names_[i])->backward(&storage_, worker_index);
    }

    std::vector<vec_t> forward(const std::vector<vec_t>& in_data, int worker_index) {

        // set input-data to first layers
        for (size_t i = 0; i < in_data.size(); i++) {
            for (size_t j = 0; j < input2data_[i].size(); j++) {
                vec_t *v = storage_.get(input2data_[i][j], worker_index);
                *v = in_data[i];
            }
        }

        // propagate to output
        for (int i = 0; i < (int)layer_names_.size(); i++)
            get_by_bame(layer_names_[i])->forward(&storage_, worker_index);

        return get_by_bame(layer_names_.back())->output(storage_, worker_index);
    }

    void update_weights(optimizer *opt, int num_workers, int batch_size) {
        for (auto l : nodes_) {
            l->update_weight(opt, &storage_, num_workers, batch_size);
        }
    }

    void set_worker_count(cnn_size_t worker) {
        storage_.set_worker_size(worker);
        for (auto l : nodes_)
            l->set_worker_count(worker);
    }

    void setup(bool reset_weight, int max_task_size) {
        for (auto l : nodes_) {
            l->setup(&storage_, reset_weight, max_task_size);
        }
    }

    void add_node(nodeptr_t node, const std::string& name) {
        layer_names_.push_back(name);
        nodes_.push_back(node);
    }

    void add_edge(const std::string& from, int from_idx, const std::string& to, int to_idx) {
        get_by_bame(from)->connect(get_by_bame(to).get(), from_idx, to_idx, &storage_);
    }

    void add_edge(const std::string& from, const std::string& to) {
        get_by_bame(from)->connect(get_by_bame(to).get(), &storage_);
    }

    void add_data_input_port(const std::string& layer_name, int port_idx, int data_idx) {
        if (input2data_.size() <= (cnn_size_t)data_idx) input2data_.resize(data_idx + 1);

        auto layer = get_by_bame(layer_name);
        input2data_[data_idx].push_back(layer->in_data_index()[port_idx]);
    }

    void add_data_output_port(const std::string& layer_name, int port_idx, int data_idx) {
        if (output2grads_.size() <= (cnn_size_t)data_idx) output2grads_.resize(data_idx + 1);

        auto layer = get_by_bame(layer_name);
        output2grads_[data_idx].push_back(layer->out_grad_index()[port_idx]);
    }

    bool empty() const { return nodes_.size() == 0; }

    template <typename Optimizer>
    void update_weights(Optimizer *o, size_t worker_size, size_t batch_size) {
        for (auto pl : nodes_)
            pl->update_weight(o, static_cast<cnn_size_t>(worker_size), batch_size);
    }

    void set_parallelize(bool parallelize) {
        for (auto pl : nodes_)
            pl->set_parallelize(parallelize);
    }

private:
    nodeptr_t get_by_bame(const std::string& name) {
        auto it = std::find(layer_names_.begin(), layer_names_.end(), name);
        return nodes_[std::distance(layer_names_.begin(), it)];
    }

    std::vector<std::string> layer_names_;

    std::vector<std::vector<int>> input2data_;
    std::vector<std::vector<int>> output2grads_;

    data_storage storage_;
};

} // namespace tiny_cnn
