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
#include <memory>
#include <numeric>
#include <stdio.h>
#include "tiny_cnn/util/util.h"
#include "tiny_cnn/util/product.h"
#include "tiny_cnn/util/image.h"
#include "tiny_cnn/util/weight_init.h"
#include "tiny_cnn/optimizers/optimizer.h"

#include "tiny_cnn/activations/activation_function.h"

namespace tiny_cnn {

class layer_base;
class node;
inline void connect_node(std::shared_ptr<node> head, std::shared_ptr<node> tail, cnn_size_t head_index, cnn_size_t tail_index);
inline void connect(std::shared_ptr<layer_base> head, std::shared_ptr<layer_base> tail, cnn_size_t head_index, cnn_size_t tail_index);

enum class node_type {
   layer,
   data
};

class node {
public:
    node(node_type ntype) : ntype_(ntype), in_fixed_(false), out_fixed_(false) {}
    node(node_type ntype, cnn_size_t in_size, cnn_size_t out_size)
    : ntype_(ntype), in_fixed_(true), out_fixed_(true), prev_(in_size), next_(out_size) {}

    virtual bool is_layer() const { return ntype_ == node_type::layer; }
    virtual bool is_data() const { return ntype_ == node_type::data; }

    const std::vector<std::shared_ptr<node>>& prev() const { return prev_; }
    const std::vector<std::shared_ptr<node>>& next() const { return next_; }

protected:
    node() = delete;
    node_type ntype_;
    bool in_fixed_;
    bool out_fixed_;
    friend void connect_node(std::shared_ptr<node> head, std::shared_ptr<node> tail, cnn_size_t head_index, cnn_size_t tail_index);
    friend void connect(std::shared_ptr<layer_base> head, std::shared_ptr<layer_base> tail, cnn_size_t head_index, cnn_size_t tail_index);

    mutable std::vector<std::shared_ptr<node>> prev_;
    mutable std::vector<std::shared_ptr<node>> next_;
};

class data_node : public node {
public:
    data_node(cnn_size_t size, bool has_grad = false, bool worker_specific_data = false, bool worker_specific_grad = false)
      : node(node_type::data), worker_specific_data_(worker_specific_data), worker_specific_grad_(worker_specific_grad), data_(1, vec_t(size)) {
          if (has_grad) grad_.resize(1, vec_t(size));
      }
    data_node(const shape3d& shape, vector_type vtype)
        : node(node_type::data), worker_specific_data_(!is_trainable_weight(vtype)), worker_specific_grad_(true), data_(1, vec_t(shape.size()))
    {
        grad_.resize(1, vec_t(shape.size()));
    }

    void merge_grads(cnn_size_t worker_size, vec_t *dst) {
        *dst = grad_[0];

        for (cnn_size_t i = 1; i < worker_size; i++)
            vectorize::reduce<float_t>(&grad_[i][0], dst->size(), &(*dst)[0]);
    }

    void clear_grads(cnn_size_t worker_size) {
        for (cnn_size_t i = 0; i < worker_size; i++)
            std::fill(grad_[i].begin(), grad_[i].end(), (float_t)0);
    }

    void         set_worker_size(cnn_size_t size) {
        if (worker_specific_data_) data_.resize(size, data_[0]);
        if (worker_specific_grad_) grad_.resize(size, grad_[0]);
    }
    vec_t*       get_data(cnn_size_t worker_index = 0)     { return worker_specific_data_ ? &data_[worker_index] : &data_[0]; }
    vec_t*       get_gradient(cnn_size_t worker_index = 0) { return worker_specific_grad_ ? &grad_[worker_index] : &grad_[0]; }
    const vec_t* get_data(cnn_size_t worker_index = 0)     const { return worker_specific_data_ ? &data_[worker_index] : &data_[0]; }
    const vec_t* get_gradient(cnn_size_t worker_index = 0) const { return worker_specific_grad_ ? &grad_[worker_index] : &grad_[0]; }
private:
    bool worker_specific_data_;
    bool worker_specific_grad_;
    std::vector<vec_t> data_;
    std::vector<vec_t> grad_;
};

/**
 * base class of all kind of NN layers
 *
 * sub-class should override these methods:
 * - forward_propagation ... body of forward-pass calculation
 * - back_propagation    ... body of backward-pass calculation
 * - in_shape            ... specify input data shapes
 * - out_shape           ... specify output data shapes
 * - layer_type          ... name of layer
 **/
class layer_base : public node {
public:
    friend void connection_mismatch(const layer_base& from, const layer_base& to);

    virtual ~layer_base() = default;

    /**
     * construct N-input, M-output layer
     * @param in_type[N] type of input vector (data, weight, bias...)
     * @param out_type[M] type of output vector
     **/
    layer_base(const std::vector<vector_type>& in_type, const std::vector<vector_type>& out_type)
        : node(node_type::layer, in_type.size(), out_type.size()), parallelize_(true), in_channels_(in_type.size()), out_channels_(out_type.size()),
          in_type_(in_type), out_type_(out_type)
    {
        weight_init_ = std::make_shared<weight_init::xavier>();
        bias_init_ = std::make_shared<weight_init::constant>();
    }

    layer_base(const layer_base&) = default;
    layer_base &operator =(const layer_base&) = default;

#if !defined(_MSC_VER) || (_MSC_VER >= 1900) // default generation of move constructor is unsupported in VS2013
    layer_base(layer_base&&) = default;
    layer_base &operator = (layer_base&&) = default;
#endif

    void set_parallelize(bool parallelize) {
        parallelize_ = parallelize;
    }

    /////////////////////////////////////////////////////////////////////////
    // getter

    cnn_size_t in_channels() const { return in_channels_; }
    cnn_size_t out_channels() const { return out_channels_; }

    cnn_size_t in_data_size() const {
        return sumif(in_shape(), [&](int i) { return in_type_[i] == vector_type::data; }, [](const shape3d& s) { return s.size(); });
    }

    cnn_size_t out_data_size() const {
        return sumif(out_shape(), [&](int i) { return out_type_[i] == vector_type::data; }, [](const shape3d& s) { return s.size(); });
    }

    std::vector<const vec_t*> get_weights() const {
        std::vector<const vec_t*> v;
        for (cnn_size_t i = 0; i < in_channels_; i++)
            if (is_trainable_weight(in_type_[i])) v.push_back(ith_in_node(i)->get_data(0));
        return v;
    }

    std::vector<vec_t*> get_weights() {
        std::vector<vec_t*> v;
        for (cnn_size_t i = 0; i < in_channels_; i++)
            if (is_trainable_weight(in_type_[i])) v.push_back(ith_in_node(i)->get_data(0));
        return v;
    }

    std::vector<vec_t*> get_weight_grads() {
        std::vector<vec_t*> v;
        for (cnn_size_t i = 0; i < in_channels_; i++)
            if (is_trainable_weight(in_type_[i])) v.push_back(ith_in_node(i)->get_gradient(0));
        return v;
    }

    std::vector<data_node*> get_inputs() {
        std::vector<data_node*> nodes;
        for (cnn_size_t i = 0; i < in_channels_; i++) nodes.push_back(ith_in_node(i));
        return nodes;
    }

    std::vector<data_node*> get_outputs() {
        std::vector<data_node*> nodes;
        for (cnn_size_t i = 0; i < out_channels_; i++) nodes.push_back(ith_out_node(i));
        return nodes;
    }

    std::vector<const data_node*> get_outputs() const {
        std::vector<const data_node*> nodes;
        for (cnn_size_t i = 0; i < out_channels_; i++) nodes.push_back(ith_out_node(i));
        return nodes;
    }

    void set_out_grads(const vec_t* grad, cnn_size_t gnum, cnn_size_t worker_idx) {
        cnn_size_t j = 0;
        for (cnn_size_t i = 0; i < out_channels_; i++) {
            if (out_type_[i] != vector_type::data) continue;
            assert(j < gnum);
            *ith_out_node(i)->get_gradient(worker_idx) = grad[j++];
        }
    }

    void set_in_data(const vec_t* data, cnn_size_t dnum, cnn_size_t worker_idx) {
        cnn_size_t j = 0;
        for (cnn_size_t i = 0; i < in_channels_; i++) {
            if (in_type_[i] != vector_type::data) continue;
            assert(j < dnum);
            *ith_in_node(i)->get_data(worker_idx) = data[j++];
        }
    }

    std::vector<vec_t> output(int worker_index = 0) const {
        std::vector<vec_t> out;

        for (cnn_size_t i = 0; i < out_channels_; i++)
            if (out_type_[i] == vector_type::data) out.push_back(*ith_out_node(i)->get_data(worker_index));

        return out;
    }

    ///< data-storage index of input data
    //std::vector<int> in_data_index() const { return filter(connection_.in_data, [&](int i){ return connection_.in_type[i] == vector_type::data; }); }
    //std::vector<int> out_data_index() const { return filter(connection_.out_data, [&](int i) { return connection_.out_type[i] == vector_type::data; }); }
    //std::vector<int> out_grad_index() const { return filter(connection_.out_grad, [&](int i) { return connection_.out_type[i] == vector_type::data; }); }

    /**
     * return output value range
     * used only for calculating target value from label-id in final(output) layer
     * override properly if the layer is intended to be used as output layer
     **/
    virtual std::pair<float_t, float_t> out_value_range() const { return {0.0, 1.0}; }

    /**
     * array of input shapes (width x height x depth)
     **/
    virtual std::vector<shape3d> in_shape() const = 0;

    /**
     * array of output shapes (width x height x depth)
     **/
    virtual std::vector<shape3d> out_shape() const = 0;

    /**
     * name of layer, should be unique for each concrete class
     **/
    virtual std::string layer_type() const = 0;

    /**
     * number of incoming connections for each output unit
     * used only for weight/bias initialization methods which require fan-in size (e.g. xavier)
     * override if the layer has trainable weights, and scale of initialization is important
     **/
    virtual size_t fan_in_size() const { return in_shape()[0].width_; }

    /**
     * number of outgoing connections for each input unit
     * used only for weight/bias initialization methods which require fan-out size (e.g. xavier)
     * override if the layer has trainable weights, and scale of initialization is important
     **/
    virtual size_t fan_out_size() const { return out_shape()[0].width_; }

    /////////////////////////////////////////////////////////////////////////
    // setter
    template <typename WeightInit>
    layer_base& weight_init(const WeightInit& f) { weight_init_ = std::make_shared<WeightInit>(f); return *this; }

    template <typename BiasInit>
    layer_base& bias_init(const BiasInit& f) { bias_init_ = std::make_shared<BiasInit>(f); return *this; }

    template <typename WeightInit>
    layer_base& weight_init(std::shared_ptr<WeightInit> f) { weight_init_ = f; return *this; }

    template <typename BiasInit>
    layer_base& bias_init(std::shared_ptr<BiasInit> f) { bias_init_ = f; return *this; }

    /////////////////////////////////////////////////////////////////////////
    // save/load

    virtual void save(std::ostream& os) const {
        //if (is_exploded()) throw nn_error("failed to save weights because of infinite weight");
        auto all_weights = get_weights();

        for (auto& weight : all_weights)
          for (auto w : *weight)
            os << w <<  " ";
    }

    virtual void load(std::istream& is) {
        auto all_weights = get_weights();
        for (auto& weight : all_weights)
          for (auto& w : *weight)
            is >> w;
    }

    virtual void load(std::vector<double> src, int& idx) {
        auto all_weights = get_weights();
        for (auto& weight : all_weights)
          for (auto& w : *weight)
              w = src[idx++];
    }

    /////////////////////////////////////////////////////////////////////////
    // visualize

    ///< visualize latest output of this layer
    ///< default implementation interpret output as 1d-vector,
    ///< so "visual" layer(like convolutional layer) should override this for better visualization.
    virtual image<> output_to_image(size_t channel = 0, size_t worker_index = 0) const {
        const vec_t* output = get_outputs()[channel]->get_data(worker_index);
        return vec2image<unsigned char>(*output);
    }

    /////////////////////////////////////////////////////////////////////////
    // fprop/bprop

    /**
     * @param worker_index id of current worker-task
     * @param in_data      input vectors of this layer (data, weight, bias)
     * @param out_data     output vectors
     **/
    virtual void forward_propagation(cnn_size_t worker_index,
                                     const std::vector<vec_t*>& in_data,
                                     std::vector<vec_t*>& out_data) = 0;

    /**
     * return delta of previous layer (delta=\frac{dE}{da}, a=wx in fully-connected layer)
     * @param worker_index id of current worker-task
     * @param in_data      input vectors (same vectors as forward_propagation)
     * @param out_data     output vectors (same vectors as forward_propagation)
     * @param out_grad     gradient of output vectors (i-th vector correspond with out_data[i])
     * @param in_grad      gradient of input vectors (i-th vector correspond with in_data[i])
     **/
    virtual void back_propagation(cnn_size_t                worker_index,
                                  const std::vector<vec_t*>& in_data,
                                  const std::vector<vec_t*>& out_data,
                                  std::vector<vec_t*>&       out_grad,
                                  std::vector<vec_t*>&       in_grad) = 0;

    /**
     * return delta2 of previous layer (delta2=\frac{d^2E}{da^2}, diagonal of hessian matrix)
     * it is never called if optimizer is hessian-free
     **/
    //virtual void back_propagation_2nd(const std::vector<vec_t>& delta_in) = 0;
 
    // called afrer updating weight
    virtual void post_update() {}

    /**
     * notify changing context (train <=> test)
     **/
     virtual void set_context(net_phase ctx) { CNN_UNREFERENCED_PARAMETER(ctx); }

     std::vector<vec_t> forward(const std::vector<vec_t>& input) { // for test
         set_in_data(&input[0], input.size(), 0);
         forward(0);
         return output(0);
     }

     void forward(int worker_index) {
         std::vector<vec_t*> in_data, out_data;

         // organize input/output vectors from storage
         for (cnn_size_t i = 0; i < in_channels_; i++)
             in_data.push_back(ith_in_node(i)->get_data(worker_index));

         for (cnn_size_t i = 0; i < out_channels_; i++)
             out_data.push_back(ith_out_node(i)->get_data(worker_index));

         forward_propagation(worker_index, in_data, out_data);
     }

     void backward(int worker_index) {
         std::vector<vec_t*> in_data, out_data, in_grad, out_grad;

         // organize input/output vectors from storage
         for (cnn_size_t i = 0; i < in_channels_; i++)
             in_data.push_back(ith_in_node(i)->get_data(worker_index));

         for (cnn_size_t i = 0; i < out_channels_; i++)
             out_data.push_back(ith_out_node(i)->get_data(worker_index));

         for (cnn_size_t i = 0; i < in_channels_; i++)
             in_grad.push_back(ith_in_node(i)->get_gradient(worker_index));

         for (cnn_size_t i = 0; i < out_channels_; i++)
             out_grad.push_back(ith_out_node(i)->get_gradient(worker_index));

         back_propagation(worker_index, in_data, out_data, out_grad, in_grad);
     }

     // allocate & reset weight
     void setup(bool reset_weight, int max_task_size = CNN_TASK_SIZE) {
        if (in_shape().size() != in_channels_ || out_shape().size() != out_channels_)
            throw nn_error("");

        set_worker_count(max_task_size);
        if (reset_weight) init_weight();
     }

     void init_weight() {
         for (cnn_size_t i = 0; i < in_channels_; i++) {

             switch (in_type_[i]) {
             case vector_type::weight:
                 weight_init_->fill(ith_in_node(i)->get_data(), fan_in_size(), fan_out_size());
                 break;
             case vector_type::bias:
                 bias_init_->fill(ith_in_node(i)->get_data(), fan_in_size(), fan_out_size());
                 break;
             default:
                 break;
             }
         }
     }

     void update_weight(optimizer *o, cnn_size_t worker_size, cnn_size_t batch_size) {

        for (size_t i = 0; i < in_type_.size(); i++) {
            if (is_trainable_weight(in_type_[i])) {
                vec_t diff;
                vec_t& target = *ith_in_node(i)->get_data();

                ith_in_node(i)->merge_grads(worker_size, &diff);
                std::transform(diff.begin(), diff.end(), diff.begin(), [&](float_t x) { return x / batch_size; });
                o->update(diff, target);

                ith_in_node(i)->clear_grads(worker_size);
            }
        }
        post_update();
     }

     virtual void set_worker_count(cnn_size_t worker_count) {
         for (cnn_size_t i = 0; i < in_channels_; i++)
             ith_in_node(i)->set_worker_size(worker_count);

         for (cnn_size_t i = 0; i < out_channels_; i++)
             ith_out_node(i)->set_worker_size(worker_count);
     }

     bool has_same_weights(const layer_base& rhs, float_t eps) const {
         auto w1 = get_weights();
         auto w2 = rhs.get_weights();
         if (w1.size() != w2.size()) return false;

         for (size_t i = 0; i < w1.size(); i++) {
             if (w1[i]->size() != w2[i]->size()) return false;

             for (size_t j = 0; j < w1[i]->size(); j++)
                 if (std::abs(w1[i]->at(j) - w2[i]->at(j)) > eps) return false;
         }
         return true;
     }

protected:
    bool parallelize_;
    cnn_size_t in_channels_; // number of input vectors
    cnn_size_t out_channels_; // number of output vectors
    std::vector<vector_type> in_type_;
    std::vector<vector_type> out_type_;
 
private:
    std::shared_ptr<weight_init::function> weight_init_;
    std::shared_ptr<weight_init::function> bias_init_;

    void alloc_input(cnn_size_t i) const {
        prev_[i] = std::make_shared<data_node>(in_shape()[i], in_type_[i]);
    }

    void alloc_output(cnn_size_t i) const {
        next_[i] = std::make_shared<data_node>(out_shape()[i], out_type_[i]);
    }

    data_node*       ith_in_node(cnn_size_t i)       {
        if (!prev_[i]) alloc_input(i);
        return dynamic_cast<data_node*>(prev()[i].get());
    }
    const data_node* ith_in_node(cnn_size_t i) const {
        if (!prev_[i]) alloc_input(i);
        return dynamic_cast<const data_node*>(prev()[i].get());
    }
    data_node*       ith_out_node(cnn_size_t i)       {
        if (!next_[i]) alloc_output(i);
        return dynamic_cast<data_node*>(next()[i].get());
    }
    const data_node* ith_out_node(cnn_size_t i) const {
        if (!next_[i]) alloc_output(i);
        return dynamic_cast<const data_node*>(next()[i].get()); 
    }
};

/**
 * single-input, single-output network with activation function
 **/
template<typename Activation>
class feedforward_layer : public layer_base {
public:
    explicit feedforward_layer(const std::vector<vector_type>& in_data_type)
        : layer_base(in_data_type, std_output_order(true)) {}
    activation::function& activation_function() { return h_; }
    std::pair<float_t, float_t> out_value_range() const override { return h_.scale(); }

protected:

    void backward_activation(const vec_t& prev_delta, const vec_t& this_out, vec_t& curr_delta) {
        if (h_.one_hot()) {
            for (cnn_size_t c = 0; c < prev_delta.size(); c++) {
                curr_delta[c] = prev_delta[c] * h_.df(this_out[c]);
            }
        }
        else {
            for (cnn_size_t c = 0; c < prev_delta.size(); c++) {
                vec_t df = h_.df(this_out, c);
                curr_delta[c] = vectorize::dot(&prev_delta[0], &df[0], prev_delta.size());
            }
        }
    }

    Activation h_;
};

inline void connect_node(std::shared_ptr<node> head, std::shared_ptr<node> tail, cnn_size_t head_index = 0, cnn_size_t tail_index = 0) {
    if (head->out_fixed_) {
        head->next_[head_index] = tail;
    }
    else {
        head->next_.push_back(tail);
    }

    if (tail->in_fixed_) {
        tail->prev_[tail_index] = head;
    }
    else {
        tail->prev_.push_back(head);
    }
}

inline void connect(std::shared_ptr<layer_base> head, std::shared_ptr<layer_base> tail, cnn_size_t head_index = 0, cnn_size_t tail_index = 0) {
    auto out_shape = head->out_shape()[head_index];
    auto in_shape = tail->in_shape()[tail_index];
    assert(out_shape.size() == in_shape.size());

    if (!head->next_[head_index] && !tail->prev_[tail_index]) {
        auto newnode = std::make_shared<data_node>(out_shape.size(), true, true);
        //head->next_[head_index] = tail->prev_[tail_index] = newnode;
        connect_node(head, newnode, head_index, 0);
        connect_node(newnode, tail, 0, tail_index);
    } else if (!head->next_[head_index]) {
        head->next_[head_index] = tail->prev_[tail_index];
    } else {
        tail->prev_[tail_index] = head->next_[head_index];
    }
}

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

struct node_tuple {
    node_tuple(std::shared_ptr<layer_base> l1, std::shared_ptr<layer_base> l2) {
        nodes_.push_back(l1); nodes_.push_back(l2);
    }
    std::vector<std::shared_ptr<layer_base>> nodes_;
};

node_tuple operator , (std::shared_ptr<layer_base> l1, std::shared_ptr<layer_base> l2) {
    return node_tuple(l1, l2);
}

node_tuple operator , (node_tuple& lhs, std::shared_ptr<layer_base> rhs) {
    lhs.nodes_.push_back(rhs);
    return lhs;
}

template <typename T, typename U>
inline std::shared_ptr<T>& operator << (std::shared_ptr<T>& lhs, std::shared_ptr<U>& rhs) {
    connect(lhs, rhs);
    return lhs;
}

template <typename T>
inline const node_tuple& operator << (const node_tuple& lhs, std::shared_ptr<T>& rhs) {
    for (size_t i = 0; i < lhs.nodes_.size(); i++)
        connect(lhs.nodes_[i], rhs, 0, i);
    return lhs;
}


template <typename T>
inline std::shared_ptr<T>& operator << (std::shared_ptr<T>& lhs, const node_tuple& rhs) {
    for (size_t i = 0; i < rhs.nodes_.size(); i++)
        connect(lhs, rhs.nodes_[i], i, 0);
    return lhs;
}

// error message functions

inline void connection_mismatch(const layer_base& from, const layer_base& to) {
    std::ostringstream os;

    os << std::endl;
    os << "output size of Nth layer must be equal to input of (N+1)th layer" << std::endl;
    os << "layerN:   " << std::setw(12) << from.layer_type() << " in:" << from.in_data_size() << "(" << from.in_shape() << "), " << 
                                                "out:" << from.out_data_size() << "(" << from.out_shape() << ")" << std::endl;
    os << "layerN+1: " << std::setw(12) << to.layer_type() << " in:" << to.in_data_size() << "(" << to.in_shape() << "), " <<
                                             "out:" << to.out_data_size() << "(" << to.out_shape() << ")" << std::endl;
    os << from.out_data_size() << " != " << to.in_data_size() << std::endl;
    std::string detail_info = os.str();

    throw nn_error("layer dimension mismatch!" + detail_info);
}

inline void data_mismatch(const layer_base& layer, const vec_t& data) {
    std::ostringstream os;

    os << std::endl;
    os << "data dimension:    " << data.size() << std::endl;
    os << "network dimension: " << layer.in_data_size() << "(" << layer.layer_type() << ":" << layer.in_shape() << ")" << std::endl;

    std::string detail_info = os.str();

    throw nn_error("input dimension mismath!" + detail_info);
}

inline void pooling_size_mismatch(cnn_size_t in_width, cnn_size_t in_height, cnn_size_t pooling_size) {
    std::ostringstream os;

    os << std::endl;
    os << "WxH:" << in_width << "x" << in_height << std::endl;
    os << "pooling-size:" << pooling_size << std::endl;

    std::string detail_info = os.str();

    throw nn_error("width/height must be multiples of pooling size" + detail_info);
}

} // namespace tiny_cnn
