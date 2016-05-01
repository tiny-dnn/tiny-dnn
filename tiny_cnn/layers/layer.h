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
#include "tiny_cnn/data_storage.h"
#include "tiny_cnn/optimizers/optimizer.h"

#include "tiny_cnn/activations/activation_function.h"

namespace tiny_cnn {


enum class vector_type : int32_t {
    // 0x0001XXX : in/out data
    data   = 0x0001000, // input/output data, fed by other layer or input channel

    // 0x0002XXX : trainable parameters, updated for each back propagation
    weight = 0x0002000,
    bias   = 0x0002001,

    label  = 0x0004000,
    aux    = 0x0010000 // layer-specific storage
};

inline vector_type operator & (vector_type lhs, vector_type rhs) {
    return (vector_type)(static_cast<int32_t>(lhs) & static_cast<int32_t>(rhs));
}

inline bool is_trainable_weight(vector_type vtype) {
    return (vtype & vector_type::weight) == vector_type::weight;
}

inline std::vector<vector_type> std_input_order(bool has_bias) {
    if (has_bias) {
        return { vector_type::data, vector_type::weight, vector_type::bias };
    }
    else {
        return { vector_type::data, vector_type::weight };
    }
}

inline std::vector<vector_type> std_output_order(bool has_activation) {
    if (has_activation) {
        return { vector_type::data, vector_type::aux };
    }
    else {
        return { vector_type::data };
    }
}

/**
 * base class of all kind of NN layers
 **/
class layer_base {
public:
    friend void connection_mismatch(const layer_base& from, const layer_base& to);

    virtual ~layer_base() = default;

    /**
     * construct N-input, M-output layer
     * @param in_type[N] type of input vector (data, weight, bias...)
     * @param out_type[M] type of output vector
     **/
    layer_base(const std::vector<vector_type>& in_type, const std::vector<vector_type>& out_type)
        : parallelize_(true), in_channels_(in_type.size()), out_channels_(out_type.size()),
          connection_(in_type, out_type)
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

    ///< connect this output[idx] and other's input[idx]
    void connect(layer_base* other, cnn_size_t this_idx, cnn_size_t other_idx, data_storage *storage) {
        auto this_shape = out_shape()[this_idx];
        auto other_shape = other->in_shape()[other_idx];

        if (this_shape.size() != other_shape.size())
            throw nn_error("dimension mismatch");

        // @todo refactoring
        if (connection_.out_data[this_idx] == -1 && other->connection_.in_data[other_idx] == -1) {
            connection_.out_data[this_idx] = other->connection_.in_data[other_idx] = storage->allocate(out_shape()[this_idx], true);
            connection_.out_grad[this_idx] = other->connection_.in_grad[other_idx] = storage->allocate(out_shape()[this_idx], true);
        }
        else if (connection_.out_data[this_idx] == -1) {
            connection_.out_data[this_idx] = other->connection_.in_data[other_idx];
            connection_.out_grad[this_idx] = other->connection_.in_grad[other_idx];
        }
        else if (other->connection_.in_data[other_idx] == -1) {
            other->connection_.in_data[other_idx] = connection_.out_data[this_idx];
            other->connection_.in_grad[other_idx] = connection_.out_grad[this_idx];
        }
        else if (connection_.out_data[this_idx] != other->connection_.in_data[other_idx]) {
            throw nn_error("connection duplicated");
        }
    }

    void connect(layer_base* other, data_storage *storage) {
        if (out_channels() < other->in_channels())
            throw nn_error("channels mismatch");

        for (cnn_size_t i = 0; i < other->in_channels(); i++)
            connect(other, i, i, storage);
    }

    void connect_input(int in_idx, int data_id, int grad_id) {
        connection_.in_data[in_idx] = data_id;
        connection_.in_grad[in_idx] = grad_id;
    }

    void connect_output(int out_idx, int data_id, int grad_id) {
        connection_.out_data[out_idx] = data_id;
        connection_.out_grad[out_idx] = grad_id;
    }

    /////////////////////////////////////////////////////////////////////////
    // getter

    cnn_size_t in_channels() const { return in_channels_; }
    cnn_size_t out_channels() const { return out_channels_; }

    cnn_size_t in_data_size() const {
        return sumif(in_shape(), [&](int i) { return connection_.in_type[i] == vector_type::data; }, [](const shape3d& s) { return s.size(); });
    }

    cnn_size_t out_data_size() const {
        return sumif(out_shape(), [&](int i) { return connection_.out_type[i] == vector_type::data; }, [](const shape3d& s) { return s.size(); });
    }

    std::vector<const vec_t*> get_weights(const data_storage& storage) const {
        std::vector<const vec_t*> vec = storage.get(connection_.in_data);
        return filter(vec, [&](int i){ return is_trainable_weight(connection_.in_type[i]); });
    }

    std::vector<vec_t*> get_weights(data_storage& storage) {
        std::vector<vec_t*> vec = storage.get(connection_.in_data);
        return filter(vec, [&](int i) { return is_trainable_weight(connection_.in_type[i]); });
    }

    std::vector<const vec_t*> get_grads(const data_storage& storage) const {
        std::vector<const vec_t*> vec = storage.get(connection_.in_grad);
        return filter(vec, [&](int i) { return is_trainable_weight(connection_.in_type[i]); });
    }

    std::vector<vec_t> output(const data_storage& storage, int worker_index = 0) const {
        std::vector<vec_t> out;

        for (cnn_size_t i = 0; i < out_channels_; i++) {
            if (connection_.out_type[i] == vector_type::data) out.push_back(*storage.get(connection_.out_data[i], worker_index));
        }
        return out;
    }

    ///< data-storage index of input data
    std::vector<int> in_data_index() const { return filter(connection_.in_data, [&](int i){ return connection_.in_type[i] == vector_type::data; }); }
    std::vector<int> out_data_index() const { return filter(connection_.out_data, [&](int i) { return connection_.out_type[i] == vector_type::data; }); }
    std::vector<int> out_grad_index() const { return filter(connection_.out_grad, [&](int i) { return connection_.out_type[i] == vector_type::data; }); }

    ///< output value range
    ///< used for calculating target value from label
    virtual std::pair<float_t, float_t> out_value_range() const { return {0.0, 1.0}; }

    ///< input shape(width x height x depth)
    virtual std::vector<shape3d> in_shape() const = 0;

    ///< output shape(width x height x depth)
    virtual std::vector<shape3d> out_shape() const = 0;

    ///< name of layer. should be unique for each concrete class
    virtual std::string layer_type() const = 0;

    ///< number of incoming connections for each output unit
    virtual size_t fan_in_size() const = 0;

    ///< number of outgoing connections for each input unit
    virtual size_t fan_out_size() const = 0;

    virtual size_t connection_size() const = 0;

    /////////////////////////////////////////////////////////////////////////
    // save/load

    virtual void save(std::ostream& os, const data_storage& storage) const {
        //if (is_exploded()) throw nn_error("failed to save weights because of infinite weight");
        auto all_weights = get_weights(storage);

        for (auto& weight : all_weights)
          for (auto w : *weight)
            os << w <<  " ";
    }

    virtual void load(std::istream& is, data_storage& storage) {
        auto all_weights = get_weights(storage);
        for (auto& weight : all_weights)
          for (auto& w : *weight)
            is >> w;
    }

    virtual void load(std::vector<double> src, int& idx, data_storage& storage) {
        auto all_weights = get_weights(storage);
        for (auto& weight : all_weights)
          for (auto& w : *weight)
              w = src[idx++];
    }

    /////////////////////////////////////////////////////////////////////////
    // visualize

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

     void forward(data_storage *storage, int worker_index) {
         std::vector<vec_t*> in_data, out_data;

         // organize input/output vectors from storage
         prepare(connection_.in_data,  storage, worker_index, &in_data);
         prepare(connection_.out_data, storage, worker_index, &out_data);

         forward_propagation(worker_index, in_data, out_data);
     }

     void backward(data_storage *storage, int worker_index) {
         std::vector<vec_t*> in_data, out_data, in_grad, out_grad;

         // organize input/output vectors from storage
         prepare(connection_.in_data, storage, worker_index, &in_data);
         prepare(connection_.out_data, storage, worker_index, &out_data);
         prepare(connection_.in_grad, storage, worker_index, &in_grad);
         prepare(connection_.out_grad, storage, worker_index, &out_grad);

         back_propagation(worker_index, in_data, out_data, out_grad, in_grad);
     }

     // allocate & reset weight
     void setup(data_storage *storage, bool reset_weight, int max_task_size) {
        if (in_shape().size() != in_channels_ || out_shape().size() != out_channels_)
            throw nn_error("");

        storage->set_worker_size(max_task_size);
        allocate_data(in_shape(), connection_.in_data, connection_.in_type, storage);
        allocate_grad(in_shape(), connection_.in_grad, connection_.in_type, storage);
        allocate_data(out_shape(), connection_.out_data, connection_.out_type, storage);
        allocate_grad(out_shape(), connection_.out_grad, connection_.out_type, storage);

        init_weight(storage);
     }

     void init_weight(data_storage *storage) {
         for (cnn_size_t i = 0; i < in_channels_; i++) {
             int index = connection_.in_data[i];
             switch (connection_.in_type[i]) {
             case vector_type::weight:
                 storage->foreach(index, [&](vec_t* v) { weight_init_->fill(v, fan_in_size(), fan_out_size()); });
                 break;
             case vector_type::bias:
                 storage->foreach(index, [&](vec_t* v) { bias_init_->fill(v, fan_in_size(), fan_out_size()); });
                 break;
             default:
                 break;
             }
         }
     }

     void update_weight(optimizer *o, data_storage *storage, cnn_size_t worker_size, cnn_size_t batch_size) {
        size_t in_size = connection_.in_data.size();

        for (size_t i = 0; i < in_size; i++) {
            if (is_trainable_weight(connection_.in_type[i])) {
                vec_t diff;
                vec_t& target = *storage->get(connection_.in_data[i]);

                storage->merge(connection_.in_grad[i], worker_size, &diff);
                std::transform(diff.begin(), diff.end(), diff.begin(), [&](float_t x) { return x / batch_size; });

                o->update(diff, target);
            }
            storage->clear(connection_.in_grad[i], worker_size);
        }
        post_update();
     }

     virtual void set_worker_count(cnn_size_t worker_count) {

     }

protected:
    bool parallelize_;
    cnn_size_t in_channels_; // number of input vectors
    cnn_size_t out_channels_; // number of output vectors

    struct connection {
        connection(const std::vector<vector_type>& in_type, const std::vector<vector_type>& out_type)
            : in_data(in_type.size(), -1), out_data(out_type.size(), -1),
              in_grad(in_type.size(), -1), out_grad(out_type.size(), -1), in_type(in_type), out_type(out_type) {}
        std::vector<int> in_data; // data index of storage
        std::vector<int> out_data; // data index of storage
        std::vector<int> in_grad;
        std::vector<int> out_grad;
        std::vector<vector_type> in_type;
        std::vector<vector_type> out_type;
    };
    connection connection_;

private:
    std::shared_ptr<weight_init::function> weight_init_;
    std::shared_ptr<weight_init::function> bias_init_;

    void prepare(const std::vector<int>& data_idx, data_storage *storage, int worker_index, std::vector<vec_t*> *dst) {
        for (size_t i = 0; i < data_idx.size(); i++)
            dst->push_back(storage->get(data_idx[i], worker_index));
    }

    void allocate_data(const std::vector<index3d<cnn_size_t>>& shape, std::vector<int>& connections, const std::vector<vector_type>& type, data_storage *storage) {
        for (cnn_size_t i = 0; i < shape.size(); i++) {
            if (connections[i] == -1)
                connections[i] = storage->allocate(shape[i], !is_trainable_weight(type[i]));
        }
    }

    void allocate_grad(const std::vector<index3d<cnn_size_t>>& shape, std::vector<int>& connections, const std::vector<vector_type>& type, data_storage *storage) {
        for (cnn_size_t i = 0; i < shape.size(); i++) {
            if (connections[i] == -1)
                connections[i] = storage->allocate(shape[i], true);
        }
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
