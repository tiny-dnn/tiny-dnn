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

#include "tiny_cnn/layers/layer.fwd.h"

namespace tiny_cnn {

layer::layer(const std::vector<vector_type>& in_type,
        const std::vector<vector_type>& out_type)
        : node(in_type.size(), out_type.size())
        , initialized_(false)
        , parallelize_(true)
        , in_channels_(in_type.size())
        , out_channels_(out_type.size())
        , in_type_(in_type)
        , out_type_(out_type) {
    weight_init_ = std::make_shared<weight_init::xavier>();
    bias_init_ = std::make_shared<weight_init::constant>();
}

void layer::set_parallelize(bool parallelize) {
    parallelize_ = parallelize;
}

void layer::set_backend(std::shared_ptr<core::backend> backend) {
    backend_ = backend;
}

// Creates a new program based on the kernel string. Note that the kernel string is moved-out when
// constructing the program to save copying: it should no longer be used in the remainder of this
// function.
#if defined(USE_OPENCL) || defined(USE_CUDA)
void layer::tune_kernel(const std::string& program_string,
                    std::vector<std::string>& compiler_options,
                    const CLCudaAPI::Context& context,
                    const CLCudaAPI::Device& device) {
    auto program = CLCudaAPI::Program(
        context, std::move(program_string));

    // Builds this program and checks for any compilation errors. If there are any, they are printed
    // and execution is halted.
    printf("## Compiling the kernel...\n");
    auto build_status = program.Build(device, compiler_options);
    if (build_status != CLCudaAPI::BuildStatus::kSuccess) {
        auto message = program.GetBuildInfo(device);
        printf(" > Compiler error(s)/warning(s) found:\n%s\n",
                message.c_str());
        return;
    }
    
    // setup op kernel
    // kernel_ = CLCudaAPI::Kernel(program, layer_type());
}
#endif  // USE_OPENCL OR USE_CUDA

#ifdef CNN_USE_LIBDNN
void layer::tune_kernel(const CLCudaAPI::Context& context,
                    const CLCudaAPI::Device& device,
                    const CLCudaAPI::Queue& queue,
                    const int id,
                    const int id_list,
                    const core::conv_params& params) {
    greentea::device::setupViennaCLContext(
            id, context(), device(), queue());

    std::shared_ptr<greentea::device> dev_ptr =
        std::make_shared<greentea::device>(
            id, id_list, greentea::Backend::BACKEND_OpenCL);

    // Initialize device pointer in libdnn
    dev_ptr->Init();

    // Setup libdnn params
    greentea::LibDNNConfig config;

    config.dev_ptr = dev_ptr.get();

    // NCHW shape setups

    const float_t dy = params.in_padded.height_ - params.in.height_;
    const float_t dx = params.in_padded.width_  - params.in.width_;

    std::vector<int32_t> in_shape = {
        1,
        params.in.depth_,
        params.in.height_,
        params.in.width_
    };

    std::vector<int32_t> out_shape = {
        1,
        params.out.depth_,
        params.out.height_,
        params.out.width_
    };

    std::vector<int32_t> kernel = {
            params.weight.height_,
            params.weight.width_
    };

    std::vector<int32_t> pad = { dy/2, dx/2 };

    std::vector<int32_t> stride = {
        params.h_stride,
        params.w_stride
    };

    std::vector<int32_t> dilation = { 1, 1 };

    config.in_shape = in_shape;
    config.out_shape = out_shape;
    config.pad = pad;
    config.kernel = kernel;
    config.stride = stride;
    config.dilation = dilation;
    config.group = 1;

    config.bias_term = params.has_bias;

    // Disables some optimizations but may give more stable results
    config.fast_unsafe_math = false;
    // Disables backward pass of weights during kernel.Backward();
    config.weights_backward = false;
    // Disables backward pass for bias during kernel.Backward();
    config.bias_backward    = false;
    // (Disabling bias and weight backward pass only propagates the data gradient (error))

    if (std::is_same<float_t, float>::value ||
        dev_ptr->CheckCapability("cl_khr_int64_base_atomics")) {
        config.wgalgo = greentea::LIBDNN_CONVOLUTION_WG_ALGO_ATOMIC;
        config.bwalgo = greentea::LIBDNN_CONVOLUTION_BW_ALGO_COL2IM_ATOMIC;
    } else {
        config.wgalgo = greentea::LIBDNN_CONVOLUTION_WG_ALGO_DIRECT;
        config.bwalgo = greentea::LIBDNN_CONVOLUTION_BW_ALGO_IM2COL;
    }

    // Generate the libdnn kernels
    // TODO(edgar): it should somehow be a parsistent attribute
    // kernel_ = greentea::LibDNNConv<float_t>(config);
}

#endif

/////////////////////////////////////////////////////////////////////////
// getter

bool layer::parallelize() const { return parallelize_; }
bool layer::initialized() const { return initialized_; }

core::backend_t layer::backend_type() const {
    return backend_->get_type();
}

std::shared_ptr<core::backend> layer::backend() {
    return backend_;
}

core::params layer::params() const {
    return params_;
}

///< number of incoming edges in this layer
cnn_size_t layer::in_channels() const {
    return in_channels_;
}

///< number of outgoing edges in this layer
cnn_size_t layer::out_channels() const {
    return out_channels_;
}

cnn_size_t layer::in_data_size() const {
    return sumif(in_shape(), [&](cnn_size_t i) { // NOLINT
        return in_type_[i] == vector_type::data; }, [](const shape3d& s) {
            return s.size(); });
}

cnn_size_t layer::out_data_size() const {
    return sumif(out_shape(), [&](cnn_size_t i) { // NOLINT
        return out_type_[i] == vector_type::data; }, [](const shape3d& s) {
            return s.size(); });
}

std::vector<shape3d> layer::in_data_shape() {
    return filter(in_shape(), [&](size_t i) { // NOLINT
        return in_type_[i] == vector_type::data;
    });
}

std::vector<shape3d> layer::out_data_shape() {
    return filter(out_shape(), [&](size_t i) { // NOLINT
        return out_type_[i] == vector_type::data;
    });
}

///! @deprecated use in_data_size() instead
cnn_size_t layer::in_size() const {
    return in_data_size();
}

///! @deprecated use out_data_size() instead
cnn_size_t layer::out_size() const {
    return out_data_size();
}

std::vector<const vec_t*> layer::weights() const {
    std::vector<const vec_t*> v;
    for (cnn_size_t i = 0; i < in_channels_; i++) {
        if (is_trainable_weight(in_type_[i])) {
            v.push_back(get_weight_data(i));
        }
    }
    return v;
}

std::vector<vec_t*> layer::weights() {
    std::vector<vec_t*> v;
    for (cnn_size_t i = 0; i < in_channels_; i++) {
        if (is_trainable_weight(in_type_[i])) {
            v.push_back(get_weight_data(i));
        }
    }
    return v;
}

std::vector<tensor_t*> layer::weight_grads() {
    std::vector<tensor_t*> v;
    for (cnn_size_t i = 0; i < in_channels_; i++) {
        if (is_trainable_weight(in_type_[i])) {
            v.push_back(ith_in_node(i)->get_gradient());
        }
    }
    return v;
}

std::vector<edgeptr_t> layer::inputs() {
    std::vector<edgeptr_t> nodes;
    for (cnn_size_t i = 0; i < in_channels_; i++) {
        nodes.push_back(ith_in_node(i));
    }
    return nodes;
}

std::vector<edgeptr_t> layer::outputs() {
    std::vector<edgeptr_t> nodes;
    for (cnn_size_t i = 0; i < out_channels_; i++) {
        nodes.push_back(ith_out_node(i));
    }
    return nodes;
}

std::vector<edgeptr_t> layer::outputs() const {
    std::vector<edgeptr_t> nodes;
    for (cnn_size_t i = 0; i < out_channels_; i++) {
        nodes.push_back(const_cast<layerptr_t>(this)->ith_out_node(i));
    }
    return nodes;
}

void layer::set_out_grads(const std::vector<tensor_t>& grad) {
    cnn_size_t j = 0;
    for (cnn_size_t i = 0; i < out_channels_; i++) {
        if (out_type_[i] != vector_type::data) continue;
        assert(j < grad.size());
        *ith_out_node(i)->get_gradient() = grad[j++];
    }
}

void layer::set_in_data(const std::vector<tensor_t>& data) {
    cnn_size_t j = 0;
    for (cnn_size_t i = 0; i < in_channels_; i++) {
        if (in_type_[i] != vector_type::data) continue;
        assert(j < data.size());
        *ith_in_node(i)->get_data() = data[j++];
    }
}

std::vector<tensor_t> layer::output() const {
    std::vector<tensor_t> out;
    for (cnn_size_t i = 0; i < out_channels_; i++) {
        if (out_type_[i] == vector_type::data) {
            out.push_back(*(const_cast<layerptr_t>(this))
                ->ith_out_node(i)->get_data());
        }
    }
    return out;
}

std::vector<vector_type> layer::in_types() const {
    return in_type_;
}

std::vector<vector_type> layer::out_types() const {
    return out_type_;
}

/**
    * return output value range
    * used only for calculating target value from label-id in final(output) layer
    * override properly if the layer is intended to be used as output layer
    **/
std::pair<float_t, float_t>
layer::out_value_range() const {
    return {float_t(0.0), float_t(1.0)};
}

/**
    * number of incoming connections for each output unit
    * used only for weight/bias initialization methods which require fan-in size (e.g. xavier)
    * override if the layer has trainable weights, and scale of initialization is important
    **/
size_t layer::fan_in_size() const {
    return in_shape()[0].width_;
}

/**
    * number of outgoing connections for each input unit
    * used only for weight/bias initialization methods which require fan-out size (e.g. xavier)
    * override if the layer has trainable weights, and scale of initialization is important
    **/
size_t layer::fan_out_size() const {
    return out_shape()[0].width_;
}

/////////////////////////////////////////////////////////////////////////
// setter
template <typename WeightInit>
layer& layer::weight_init(const WeightInit& f) {
    weight_init_ = std::make_shared<WeightInit>(f);
    return *this;
}

template <typename BiasInit>
layer& layer::bias_init(const BiasInit& f) {
    bias_init_ = std::make_shared<BiasInit>(f);
    return *this;
}

template <typename WeightInit>
layer& layer::weight_init(std::shared_ptr<WeightInit> f) {
    weight_init_ = f;
    return *this;
}

template <typename BiasInit>
layer& layer::bias_init(std::shared_ptr<BiasInit> f) {
    bias_init_ = f;
    return *this;
}

/////////////////////////////////////////////////////////////////////////
// save/load

void layer::save(std::ostream& os) const { // NOLINT
    /*if (is_exploded()) {
        throw nn_error("failed to save weights because of infinite weight");
    }*/
    auto all_weights = weights();
    for (auto& weight : all_weights) {
        for (auto w : *weight) os << w <<  " ";
    }
}

void layer::load(std::istream& is) { // NOLINT
    auto all_weights = weights();
    for (auto& weight : all_weights) {
        for (auto& w : *weight) is >> w;
    }
    initialized_ = true;
}

void layer::load(const std::vector<float_t>& src, int& idx) { // NOLINT
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
image<> layer::output_to_image(size_t channel) const {
    const vec_t* output = &(*(outputs()[channel]->get_data()))[0];
    return vec2image<unsigned char>(*output, out_shape()[channel]);
}

/////////////////////////////////////////////////////////////////////////
// fprop/bprop

// called afrer updating weight
void layer::post_update() {}

/**
* notify changing context (train <=> test)
**/
void layer::set_context(net_phase ctx) {
    CNN_UNREFERENCED_PARAMETER(ctx);
}

std::vector<tensor_t>
layer::forward(const std::vector<tensor_t>& input) {   // for test
    setup(false);
    set_in_data(input);
    forward();
    return output();
}

std::vector<tensor_t>
layer::backward(const std::vector<tensor_t>& out_grads) {   // for test
    setup(false);
    set_out_grads(out_grads);
    backward();
    return map_<tensor_t>(inputs(), [](edgeptr_t e) {
        return *e->get_gradient();
    });
}

void layer::forward() {
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

void layer::backward() {
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
void layer::setup(bool reset_weight) {
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

void layer::init_weight() {
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

void layer::clear_grads() {
    for (size_t i = 0; i < in_type_.size(); i++) {
        ith_in_node(i)->clear_grads();
    }
}

void layer::update_weight(optimizer *o, cnn_size_t batch_size) {
    float_t rcp_batch_size = float_t(1) / float_t(batch_size);
    for (size_t i = 0; i < in_type_.size(); i++) {
        if (is_trainable_weight(in_type_[i])) {
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

bool layer::has_same_weights(const layer& rhs, float_t eps) const {
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

void layer::set_sample_count(cnn_size_t sample_count) {

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

void layer::alloc_input(cnn_size_t i) const {
    // TODO(nyanp): refactoring
    // which type of refactoring do you have in mind for that?
    prev_[i] = std::make_shared<edge>(nullptr, in_shape()[i], in_type_[i]);
}

void layer::alloc_output(cnn_size_t i) const {
    // TODO(nyanp): refactoring
    // which type of refactoring do you have in mind for that?
    next_[i] = std::make_shared<edge>((layer*)this,
        out_shape()[i], out_type_[i]);
}

edgeptr_t layer::ith_in_node(cnn_size_t i) {
    if (!prev_[i]) alloc_input(i);
    return prev()[i];
}

edgeptr_t layer::ith_out_node(cnn_size_t i) {
    if (!next_[i]) alloc_output(i);
    return next()[i];
}

vec_t* layer::get_weight_data(cnn_size_t i) {
    assert(is_trainable_weight(in_type_[i]));
    return &(*(ith_in_node(i)->get_data()))[0];
}

const vec_t* layer::get_weight_data(cnn_size_t i) const {
    assert(is_trainable_weight(in_type_[i]));
    return &(*(const_cast<layerptr_t>(this)->ith_in_node(i)->get_data()))[0];
}

// END layer IMPLEMENTATION


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

}  // namespace tiny_cnn
