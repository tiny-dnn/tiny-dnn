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
#include <algorithm>
#include <vector>
#include <string>
#include <utility>
#include <queue>

#include "tiny_dnn/node.h"
#include "tiny_dnn/core/backend.h"
#include "tiny_dnn/core/framework/device.fwd.h"

#include "tiny_dnn/util/util.h"
#include "tiny_dnn/util/product.h"
#include "tiny_dnn/util/image.h"
#include "tiny_dnn/util/weight_init.h"

#include "tiny_dnn/optimizers/optimizer.h"
#include "tiny_dnn/activations/activation_function.h"

namespace tiny_dnn {

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
class layer : public node {
 public:
    friend void connection_mismatch(const layer& from,
                                    const layer& to);

    virtual ~layer() = default;

    /**
     * construct N-input, M-output layer
     * @param in_type[N] type of input vector (data, weight, bias...)
     * @param out_type[M] type of output vector
     **/
    layer(const std::vector<vector_type>& in_type,
          const std::vector<vector_type>& out_type)
            : node(in_type.size(), out_type.size()),
              initialized_(false),
              parallelize_(true),
              in_channels_(in_type.size()),
              out_channels_(out_type.size()),
              in_type_(in_type),
              out_type_(out_type) {
        weight_init_ = std::make_shared<weight_init::xavier>();
        bias_init_ = std::make_shared<weight_init::constant>();
        trainable_ = true;
    }

    layer(const layer&) = default;
    layer &operator =(const layer&) = default;

#ifdef CNN_USE_DEFAULT_MOVE_CONSTRUCTORS
    layer(layer&&) = default;
    layer &operator = (layer&&) = default;
#endif

    void set_parallelize(bool parallelize) {
        parallelize_ = parallelize;
    }

    void set_backend(std::shared_ptr<core::backend> backend) {
        backend_ = backend;
    }

    void set_backend_type(core::backend_t backend_type) {
        backend_type_ = backend_type;
    }

    /////////////////////////////////////////////////////////////////////////
    // getter

    bool parallelize() const { return parallelize_; }

    // TODO(edgar): Deprecated: use the below method 
    core::backend_t backend_type() const {
        return backend_->type();
    }

    core::backend_t engine() const {
        return backend_type_;
    }

    virtual std::string kernel_file() const {
        return std::string("empty_kernel_str");
    }

    virtual std::string kernel_header() const {
        return std::string();
    }

    virtual void createOp() {
    }

    void setDevice(const Device& device) {
        device_ptr_ = const_cast<Device*>(&device);
    }

    Device* device() const {
        return device_ptr_;
    }

    std::shared_ptr<core::backend> backend() { return backend_; }

    ///< number of incoming edges in this layer
    cnn_size_t in_channels() const { return in_channels_; }

    ///< number of outgoing edges in this layer
    cnn_size_t out_channels() const { return out_channels_; }

    cnn_size_t in_data_size() const {
        return sumif(in_shape(), [&](cnn_size_t i) { // NOLINT
            return in_type_[i] == vector_type::data; }, [](const shape3d& s) {
                return s.size(); });
    }

    cnn_size_t out_data_size() const {
        return sumif(out_shape(), [&](cnn_size_t i) { // NOLINT
            return out_type_[i] == vector_type::data; }, [](const shape3d& s) {
                return s.size(); });
    }

    std::vector<shape3d> in_data_shape() {
        return filter(in_shape(), [&](size_t i) { // NOLINT
            return in_type_[i] == vector_type::data;
        });
    }

    std::vector<shape3d> out_data_shape() {
        return filter(out_shape(), [&](size_t i) { // NOLINT
            return out_type_[i] == vector_type::data;
        });
    }

    ///! @deprecated use in_data_size() instead
    cnn_size_t in_size() const {
        return in_data_size();
    }

    ///! @deprecated use out_data_size() instead
    cnn_size_t out_size() const {
        return out_data_size();
    }

    std::vector<const vec_t*> weights() const {
        std::vector<const vec_t*> v;
        for (cnn_size_t i = 0; i < in_channels_; i++) {
            if (is_trainable_weight(in_type_[i])) {
                v.push_back(get_weight_data(i));
            }
        }
        return v;
    }

    std::vector<vec_t*> weights() {
        std::vector<vec_t*> v;
        for (cnn_size_t i = 0; i < in_channels_; i++) {
            if (is_trainable_weight(in_type_[i])) {
                v.push_back(get_weight_data(i));
            }
        }
        return v;
    }

    std::vector<tensor_t*> weights_grads() {
        std::vector<tensor_t*> v;
        for (cnn_size_t i = 0; i < in_channels_; i++) {
            if (is_trainable_weight(in_type_[i])) {
                v.push_back(ith_in_node(i)->get_gradient());
            }
        }
        return v;
    }

    std::vector<edgeptr_t> inputs() {
        std::vector<edgeptr_t> nodes;
        for (cnn_size_t i = 0; i < in_channels_; i++) {
            nodes.push_back(ith_in_node(i));
        }
        return nodes;
    }

    std::vector<edgeptr_t> outputs() {
        std::vector<edgeptr_t> nodes;
        for (cnn_size_t i = 0; i < out_channels_; i++) {
            nodes.push_back(ith_out_node(i));
        }
        return nodes;
    }

    std::vector<edgeptr_t> outputs() const {
        std::vector<edgeptr_t> nodes;
        for (cnn_size_t i = 0; i < out_channels_; i++) {
            nodes.push_back(const_cast<layerptr_t>(this)
                    ->ith_out_node(i));
        }
        return nodes;
    }

    void set_out_grads(const std::vector<tensor_t>& grad) {
        cnn_size_t j = 0;
        for (cnn_size_t i = 0; i < out_channels_; i++) {
            if (out_type_[i] != vector_type::data) continue;
            assert(j < grad.size());
            *ith_out_node(i)->get_gradient() = grad[j++];
        }
    }

    void set_in_data(const std::vector<tensor_t>& data) {
        cnn_size_t j = 0;
        for (cnn_size_t i = 0; i < in_channels_; i++) {
            if (in_type_[i] != vector_type::data) continue;
            assert(j < data.size());
            *ith_in_node(i)->get_data() = data[j++];
        }
    }

    std::vector<tensor_t> output() const {
        std::vector<tensor_t> out;
        for (cnn_size_t i = 0; i < out_channels_; i++) {
            if (out_type_[i] == vector_type::data) {
                out.push_back(*(const_cast<layerptr_t>(this))
                    ->ith_out_node(i)->get_data());
            }
        }
        return out;
    }

    std::vector<vector_type> in_types() const { return in_type_; }

    std::vector<vector_type> out_types() const { return out_type_; }

    void set_trainable(bool trainable) { trainable_ = trainable; }

    bool trainable() const { return trainable_; }

    /**
     * return output value range
     * used only for calculating target value from label-id in final(output) layer
     * override properly if the layer is intended to be used as output layer
     **/
    virtual std::pair<float_t, float_t> out_value_range() const {
        return { float_t(0.0), float_t(1.0) };
    }

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
    virtual size_t fan_in_size() const {
        return in_shape()[0].width_;
    }

    /**
     * number of outgoing connections for each input unit
     * used only for weight/bias initialization methods which require fan-out size (e.g. xavier)
     * override if the layer has trainable weights, and scale of initialization is important
     **/
    virtual size_t fan_out_size() const {
        return out_shape()[0].width_;
    }

    /////////////////////////////////////////////////////////////////////////
    // setter
    template <typename WeightInit>
    layer& weight_init(const WeightInit& f) {
        weight_init_ = std::make_shared<WeightInit>(f);
        return *this;
    }

    template <typename BiasInit>
    layer& bias_init(const BiasInit& f) {
        bias_init_ = std::make_shared<BiasInit>(f);
        return *this;
    }

    template <typename WeightInit>
    layer& weight_init(std::shared_ptr<WeightInit> f) {
        weight_init_ = f;
        return *this;
    }

    template <typename BiasInit>
    layer& bias_init(std::shared_ptr<BiasInit> f) {
        bias_init_ = f;
        return *this;
    }

    /////////////////////////////////////////////////////////////////////////
    // save/load
    template <typename Archive>
    void serialize(Archive & ar) {
        auto all_weights = weights();
        for (auto weight : all_weights) {
            ar(*weight);
        }
        initialized_ = true;
    }

    virtual void save(std::ostream& os) const { // NOLINT
        /*if (is_exploded()) {
            throw nn_error("failed to save weights because of infinite weight");
        }*/
        auto all_weights = weights();
        for (auto& weight : all_weights) {
            for (auto w : *weight) os << w <<  " ";
        }
    }

    virtual void load(std::istream& is) { // NOLINT
        auto all_weights = weights();
        for (auto& weight : all_weights) {
            for (auto& w : *weight) is >> w;
        }
        initialized_ = true;
    }

    virtual void load(const std::vector<float_t>& src, int& idx) { // NOLINT
        auto all_weights = weights();
        for (auto& weight : all_weights) {
            for (auto& w : *weight) w = src[idx++];
        }
        initialized_ = true;
    }

    /////////////////////////////////////////////////////////////////////////
    // visualize

    ///< visualize latest output of this layer
    ///< default implementation interpret output as 1d-vector,
    ///< so "visual" layer(like convolutional layer) should override this for better visualization.
    virtual image<> output_to_image(size_t channel = 0) const {
        const vec_t* output = &(*(outputs()[channel]->get_data()))[0];
        return vec2image<unsigned char>(*output, out_shape()[channel]);
    }

    /////////////////////////////////////////////////////////////////////////
    // fprop/bprop

    /**
     * @param in_data      input vectors of this layer (data, weight, bias)
     * @param out_data     output vectors
     **/
    virtual void forward_propagation(const std::vector<tensor_t*>& in_data,
                                     std::vector<tensor_t*>& out_data) = 0;

    /**
     * return delta of previous layer (delta=\frac{dE}{da}, a=wx in fully-connected layer)
     * @param in_data      input vectors (same vectors as forward_propagation)
     * @param out_data     output vectors (same vectors as forward_propagation)
     * @param out_grad     gradient of output vectors (i-th vector correspond with out_data[i])
     * @param in_grad      gradient of input vectors (i-th vector correspond with in_data[i])
     **/
    virtual void back_propagation(const std::vector<tensor_t*>& in_data,
                                  const std::vector<tensor_t*>& out_data,
                                  std::vector<tensor_t*>&       out_grad,
                                  std::vector<tensor_t*>&       in_grad) = 0;

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
    virtual void set_context(net_phase ctx) {
        CNN_UNREFERENCED_PARAMETER(ctx);
    }

    std::vector<tensor_t> forward(const std::vector<tensor_t>& input) {   // for test
        setup(false);
        set_in_data(input);
        forward();
        return output();
    }

    std::vector<tensor_t> backward(const std::vector<tensor_t>& out_grads) {   // for test
        setup(false);
        set_out_grads(out_grads);
        backward();
        return map_<tensor_t>(inputs(), [](edgeptr_t e) {
            return *e->get_gradient();
        });
    }

    void forward() {
        std::vector<tensor_t*> in_data, out_data;

        // organize input/output vectors from storage
        for (cnn_size_t i = 0; i < in_channels_; i++) {
            in_data.push_back(ith_in_node(i)->get_data());
        }

        // resize outs and stuff to have room for every input sample in the batch
        set_sample_count(in_data[0]->size());

        for (cnn_size_t i = 0; i < out_channels_; i++) {
            out_data.push_back(ith_out_node(i)->get_data());
            ith_out_node(i)->clear_grads();
        }

        forward_propagation(in_data, out_data);
    }

    void backward() {
        std::vector<tensor_t*> in_data, out_data, in_grad, out_grad;

        // organize input/output vectors from storage
        for (cnn_size_t i = 0; i < in_channels_; i++) {
            in_data.push_back(ith_in_node(i)->get_data());
        }
        for (cnn_size_t i = 0; i < out_channels_; i++) {
            out_data.push_back(ith_out_node(i)->get_data());
        }
        for (cnn_size_t i = 0; i < in_channels_; i++) {
            in_grad.push_back(ith_in_node(i)->get_gradient());
        }
        for (cnn_size_t i = 0; i < out_channels_; i++) {
            out_grad.push_back(ith_out_node(i)->get_gradient());
        }
        back_propagation(in_data, out_data, out_grad, in_grad);
    }

    // allocate & reset weight
    void setup(bool reset_weight) {
        if (in_shape().size() != in_channels_ ||
            out_shape().size() != out_channels_) {
                throw nn_error("Connection mismatch at setup layer");
        }

        for (size_t i = 0; i < out_channels_; i++) {
            if (!next_[i]) {
                next_[i] = std::make_shared<edge>(
                    this, out_shape()[i], out_type_[i]);
            }
        }

        if (reset_weight || !initialized_) {
            init_weight();
        }
    }

    void init_weight() {
        if (!trainable_) {
            initialized_ = true;
            return;
        }

        for (cnn_size_t i = 0; i < in_channels_; i++) {
            switch (in_type_[i]) {
                case vector_type::weight:
                    weight_init_->fill(get_weight_data(i),
                                       fan_in_size(), fan_out_size());
                    break;
                case vector_type::bias:
                    bias_init_->fill(get_weight_data(i),
                                     fan_in_size(), fan_out_size());
                    break;
                default:
                    break;
            }
        }
        initialized_ = true;
    }

    void clear_grads() {
        for (size_t i = 0; i < in_type_.size(); i++) {
            ith_in_node(i)->clear_grads();
        }
    }

    void update_weight(optimizer *o, cnn_size_t batch_size) {
        float_t rcp_batch_size = float_t(1) / float_t(batch_size);
        for (size_t i = 0; i < in_type_.size(); i++) {
            if (trainable() && is_trainable_weight(in_type_[i])) {
                vec_t diff;
                vec_t& target = *get_weight_data(i);

                ith_in_node(i)->merge_grads(&diff);
                std::transform(diff.begin(), diff.end(),
                               diff.begin(), [&](float_t x) { // NOLINT
                                  return x * rcp_batch_size; });
                o->update(diff, target);
            }
        }
        clear_grads();
        post_update();
    }

    bool has_same_weights(const layer& rhs, float_t eps) const {
        auto w1 = weights();
        auto w2 = rhs.weights();
        if (w1.size() != w2.size()) return false;

        for (size_t i = 0; i < w1.size(); i++) {
            if (w1[i]->size() != w2[i]->size()) return false;

            for (size_t j = 0; j < w1[i]->size(); j++) {
                if (std::abs(w1[i]->at(j) - w2[i]->at(j)) > eps) return false;
            }
        }
        return true;
    }

    virtual void set_sample_count(cnn_size_t sample_count) {

        // increase the size if necessary - but do not decrease
        auto resize = [sample_count](tensor_t* tensor) {
            tensor->resize(sample_count, (*tensor)[0]);
        };

        for (cnn_size_t i = 0; i < in_channels_; i++) {
            if (!is_trainable_weight(in_type_[i])) {
                resize(ith_in_node(i)->get_data());
            }
            resize(ith_in_node(i)->get_gradient());
        }

        for (cnn_size_t i = 0; i < out_channels_; i++) {
            if (!is_trainable_weight(out_type_[i])) {
                resize(ith_out_node(i)->get_data());
            }
            resize(ith_out_node(i)->get_gradient());
        }
    }

    /**
    * generate layer from cereal's Archive
    **/
    template <typename InputArchive>
    static std::shared_ptr<layer> load_layer(InputArchive & ia);

    template <typename OutputArchive>
    static void save_layer(OutputArchive & oa, const layer& l);

    template <class Archive>
    void serialize_prolog(Archive & ar);

 protected:
    bool initialized_;
    bool parallelize_;
    cnn_size_t in_channels_;   // number of input vectors
    cnn_size_t out_channels_;  // number of output vectors
    std::vector<vector_type> in_type_;
    std::vector<vector_type> out_type_;
    
    core::backend_t backend_type_;
    std::shared_ptr<core::backend> backend_;

    Device* device_ptr_ = nullptr;

 private:
    bool trainable_;
    std::shared_ptr<weight_init::function> weight_init_;
    std::shared_ptr<weight_init::function> bias_init_;

    void alloc_input(cnn_size_t i) const {
        // TODO(nyanp): refactoring
        // which type of refactoring do you have in mind for that?
        prev_[i] = std::make_shared<edge>(nullptr, in_shape()[i], in_type_[i]);
    }

    void alloc_output(cnn_size_t i) const {
        // TODO(nyanp): refactoring
        // which type of refactoring do you have in mind for that?
        next_[i] = std::make_shared<edge>((layer*)this,
            out_shape()[i], out_type_[i]);
    }

    edgeptr_t ith_in_node(cnn_size_t i) {
        if (!prev_[i]) alloc_input(i);
        return prev()[i];
    }

    edgeptr_t ith_out_node(cnn_size_t i) {
        if (!next_[i]) alloc_output(i);
        return next()[i];
    }

    vec_t* get_weight_data(cnn_size_t i) {
        assert(is_trainable_weight(in_type_[i]));
        return &(*(ith_in_node(i)->get_data()))[0];
    }

    const vec_t* get_weight_data(cnn_size_t i) const {
        assert(is_trainable_weight(in_type_[i]));
        return &(*(const_cast<layerptr_t>(this)->ith_in_node(i)->get_data()))[0];
    }
};

inline void connect(layerptr_t head,
                    layerptr_t tail,
                    cnn_size_t head_index = 0,
                    cnn_size_t tail_index = 0) {
    auto out_shape = head->out_shape()[head_index];
    auto in_shape = tail->in_shape()[tail_index];

    head->setup(false);

    if (out_shape.size() != in_shape.size()) {
        connection_mismatch(*head, *tail);
    }

    if (!head->next_[head_index]) {
        throw nn_error("output edge must not be null");
    }

    tail->prev_[tail_index] = head->next_[head_index];
    tail->prev_[tail_index]->add_next_node(tail);
}

inline layer& operator << (layer& lhs, layer& rhs) {
    connect(&lhs, &rhs);
    return rhs;
}

template <typename Char, typename CharTraits>
std::basic_ostream<Char, CharTraits>& operator << (
        std::basic_ostream<Char, CharTraits>& os, const layer& v) {
    v.save(os);
    return os;
}

template <typename Char, typename CharTraits>
std::basic_istream<Char, CharTraits>& operator >> (
      std::basic_istream<Char, CharTraits>& os, layer& v) {
    v.load(os);
    return os;
}

// error message functions

inline void connection_mismatch(const layer& from, const layer& to) {
    std::ostringstream os;

    os << std::endl;
    os << "output size of Nth layer must be equal to input of (N+1)th layer\n";

    os << "layerN:   " << std::setw(12) << from.layer_type() << " in:"
                                        << from.in_data_size() << "("
                                        << from.in_shape() << "), " << "out:"
                                        << from.out_data_size() << "("
                                        << from.out_shape() << ")\n";

    os << "layerN+1: " << std::setw(12) << to.layer_type() << " in:"
                                        << to.in_data_size() << "("
                                        << to.in_shape() << "), " << "out:"
                                        << to.out_data_size() << "("
                                        << to.out_shape() << ")\n";

    os << from.out_data_size() << " != " << to.in_data_size() << std::endl;
    std::string detail_info = os.str();

    throw nn_error("layer dimension mismatch!" + detail_info);
}

inline void data_mismatch(const layer& layer, const vec_t& data) {
    std::ostringstream os;

    os << std::endl;
    os << "data dimension:    " << data.size() << "\n";
    os << "network dimension: " << layer.in_data_size() << "("
                                << layer.layer_type() << ":"
                                << layer.in_shape() << ")\n";

    std::string detail_info = os.str();

    throw nn_error("input dimension mismatch!" + detail_info);
}

inline void pooling_size_mismatch(cnn_size_t in_width,
                                  cnn_size_t in_height,
                                  cnn_size_t pooling_size) {
    std::ostringstream os;

    os << std::endl;
    os << "WxH:" << in_width << "x" << in_height << std::endl;
    os << "pooling-size:" << pooling_size << std::endl;

    std::string detail_info = os.str();

    throw nn_error("width/height not multiple of pooling size" + detail_info);
}


template <typename T, typename U>
void graph_traverse(layer *root_node, T&& node_callback, U&& edge_callback) {
    std::unordered_set<layer*> visited;
    std::queue<layer*> S;

    S.push(root_node);

    while (!S.empty()) {
        layer *curr = S.front();
        S.pop();
        visited.insert(curr);

        node_callback(*curr);

        auto edges = curr->next();
        for (auto e : edges) {
            if (e != nullptr)
                edge_callback(*e);
        }

        auto prev = curr->prev_nodes();
        for (auto p : prev) {
            // TODO(nyanp): refactoring
            // which type of refactoring do you have in mind for that?
            layer* l = dynamic_cast<layer*>(p);
            if (visited.find(l) == visited.end()) {
                S.push(l);
            }
        }

        auto next = curr->next_nodes();
        for (auto n : next) {
            // TODO(nyanp): refactoring
            // which type of refactoring do you have in mind for that?
            layer* l = dynamic_cast<layer*>(n);
            if (visited.find(l) == visited.end()) {
                S.push(l);
            }
        }
    }
}



}  // namespace tiny_dnn
