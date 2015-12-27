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
#include <algorithm>
#include <memory>
#include <unordered_map>
#include "caffe.pb.h"
#include "tiny_cnn/tiny_cnn.h"

typedef tiny_cnn::layer_shape_t shape_t;

#ifdef _WIN32
#define _NOMINMAX
#include <io.h>
#ifndef O_RDONLY
#define _O_RDONLY O_RDONLY
#endif
#endif

namespace tiny_cnn {
namespace detail {

inline std::shared_ptr<tiny_cnn::weight_init::function> create_filler(const std::string& filler) {
    if (filler == "xavier") {
        return std::make_shared<tiny_cnn::weight_init::xavier>();
    }
    else if (filler == "constant") {
        return std::make_shared<tiny_cnn::weight_init::constant>();
    }
    else if (filler == "gaussian") {
        return std::make_shared<tiny_cnn::weight_init::gaussian>();
    }
    else {
        throw std::runtime_error("unsupported filler type");
    }
}

template <typename param>
inline bool get_kernel_size_2d(const param& p, tiny_cnn::layer_size_t *kernel) {
    if (p.has_kernel_w() && p.has_kernel_w()) {
        if (p.kernel_w() != p.kernel_h())
            throw std::runtime_error("unsupported kernel shape");
        *kernel = p.kernel_w();
        return true;
    }
    return false;
}

inline std::shared_ptr<tiny_cnn::layer_base> create_max_pool(int pool_size, int stride, const shape_t& bottom_shape, shape_t *top_shape)
{
    using max_pool = tiny_cnn::max_pooling_layer<tiny_cnn::activation::identity>;
    auto mp = std::make_shared<max_pool>(bottom_shape.width_, bottom_shape.height_, bottom_shape.depth_, pool_size, stride);
    *top_shape = mp->out_shape();
    return mp;
}

inline std::shared_ptr<tiny_cnn::layer_base> create_ave_pool(int pool_size, int stride, const shape_t& bottom_shape, shape_t *top_shape)
{
    using ave_pool = tiny_cnn::average_pooling_layer<tiny_cnn::activation::identity>;
    auto ap = std::make_shared<ave_pool>(bottom_shape.width_, bottom_shape.height_, bottom_shape.depth_, pool_size, stride);

    // tiny-cnn has trainable parameter in average-pooling layer
    tiny_cnn::float_t weight = 1.0 / tiny_cnn::sqr(pool_size);
    std::fill(ap->weight().begin(), ap->weight().end(), weight);
    std::fill(ap->bias().begin(), ap->bias().end(), (tiny_cnn::float_t)0.0);
    *top_shape = ap->out_shape();
    return ap;
}

inline std::shared_ptr<tiny_cnn::layer_base> create_softmax(const caffe::LayerParameter& layer, const shape_t& bottom_shape, shape_t *) {
    auto sm = std::make_shared<tiny_cnn::linear_layer<tiny_cnn::activation::softmax>>(bottom_shape.size());
    return sm;
}

inline std::shared_ptr<tiny_cnn::layer_base> create_sigmoid(const caffe::LayerParameter& layer, const shape_t& bottom_shape, shape_t *) {
    auto ce = std::make_shared<tiny_cnn::linear_layer<tiny_cnn::activation::sigmoid>>(bottom_shape.size());
    return ce;
}

inline std::shared_ptr<tiny_cnn::layer_base> create_tanh(const caffe::LayerParameter& layer, const shape_t& bottom_shape, shape_t *) {
    auto tanh = std::make_shared<tiny_cnn::linear_layer<tiny_cnn::activation::tan_h>>(bottom_shape.size());
    return tanh;
}

inline std::shared_ptr<tiny_cnn::layer_base> create_pooling(const caffe::LayerParameter& layer, const shape_t& bottom_shape, shape_t *top_shape) {
    using max_pool = tiny_cnn::max_pooling_layer<tiny_cnn::activation::identity>;
    using ave_pool = tiny_cnn::average_pooling_layer<tiny_cnn::activation::identity>;

    if (!layer.has_pooling_param())
        throw std::runtime_error("pool param missing");

    auto pool_param = layer.pooling_param();

    tiny_cnn::layer_size_t pool_size = 0;
    tiny_cnn::layer_size_t h_stride, w_stride;

    if (!get_kernel_size_2d(pool_param, &pool_size))
        pool_size = pool_param.kernel_size();

    if (pool_param.has_stride() || pool_param.has_stride_h())
        h_stride = pool_param.has_stride() ? pool_param.stride() : pool_param.stride_h();

    if (pool_param.has_stride() || pool_param.has_stride_w())
        w_stride = pool_param.has_stride() ? pool_param.stride() : pool_param.stride_w();

    if (h_stride != w_stride)// || h_stride != pool_size)
        throw std::runtime_error("unsupported pool shape");

    if (pool_param.has_pool()) {
        auto type = pool_param.pool();

        switch (type) {
        case caffe::PoolingParameter_PoolMethod_MAX: return create_max_pool(pool_size, h_stride, bottom_shape, top_shape);
        case caffe::PoolingParameter_PoolMethod_AVE: return create_ave_pool(pool_size, h_stride, bottom_shape, top_shape);
        default: throw std::runtime_error("unsupported layer type");
        }
    }
    // default:max-pool
    return create_max_pool(pool_size, h_stride, bottom_shape, top_shape);
}

inline std::shared_ptr<tiny_cnn::layer_base> create_relu(const caffe::LayerParameter& layer, const shape_t& bottom_shape, shape_t *) {
    auto relu = std::make_shared<tiny_cnn::linear_layer<tiny_cnn::activation::relu>>(bottom_shape.size());
    return relu;
}


inline void load_weights_fullyconnected(const caffe::LayerParameter& src, tiny_cnn::layer_base *dst)
{
    auto weights = src.blobs(0);
    int curr = 0;

    if (dst->out_size() * dst->in_size() != weights.data_size())
        throw std::runtime_error(std::string("layer size mismatch!") +
            "caffe(" + src.name() + "):" + std::to_string(weights.data_size()) + "\n" +
            "tiny-cnn(" + dst->layer_type() + "):" + std::to_string(dst->weight().size()));

    for (size_t o = 0; o < dst->out_size(); o++)
        for (size_t i = 0; i < dst->in_size(); i++)
            dst->weight()[i * dst->out_size() + o] = weights.data(curr++); // transpose

                                                                        // fill bias
    if (src.inner_product_param().bias_term()) {
        auto biases = src.blobs(1);
        for (size_t o = 0; o < dst->out_size(); o++)
            dst->bias()[o] = biases.data(o);
    }
}

inline std::shared_ptr<tiny_cnn::layer_base> create_fullyconnected(const caffe::LayerParameter& layer, const shape_t& bottom_shape, shape_t *top_shape) {
    using fc_layer = tiny_cnn::fully_connected_layer<tiny_cnn::activation::identity>;

    if (!layer.has_inner_product_param())
        throw std::runtime_error("inner-product param missing");

    tiny_cnn::layer_size_t dim_input = 0, dim_output = 0;
    bool has_bias = true;

    auto ip_param = layer.inner_product_param();
    has_bias = ip_param.bias_term();

    dim_output = ip_param.num_output();
    dim_input = bottom_shape.size();

    auto ip = std::make_shared<fc_layer>(dim_input, dim_output, has_bias);

    // filler
    if (ip_param.has_weight_filler())
        ip->weight_init(create_filler(ip_param.weight_filler().type()));

    if (ip_param.has_bias_filler())
        ip->bias_init(create_filler(ip_param.bias_filler().type()));

    // weight
    if (layer.blobs_size() > 0) {
        load_weights_fullyconnected(layer, ip.get());
    }
    *top_shape = ip->out_shape();
    return ip;
}

inline void load_weights_conv(const caffe::LayerParameter& src, tiny_cnn::layer_base *dst)
{
    // fill weight
    auto weights = src.blobs(0);

    /*for (size_t o = 0; o < out_channels; o++)
        for (size_t i = 0; i < in_channels; i++)
            for (size_t y = 0; y < window_size; y++)
                for (size_t x = 0; x < window_size; x++)
                    conv->weight_at(i, o, x, y) = weights.data(curr++);
                    */

    for (size_t o = 0; o < weights.data_size(); o++) {
        dst->weight()[o] = weights.data(o);
    }

    // fill bias
    if (src.convolution_param().bias_term()) {
        auto biases = src.blobs(1);
        for (size_t o = 0; o < biases.data_size(); o++)
            dst->bias()[o] = biases.data(o);
    }
}

inline void load_weights_pool(const caffe::LayerParameter& src, tiny_cnn::layer_base *dst)
{
    auto pool_param = src.pooling_param();

    if (dst->weight().size()) {
        tiny_cnn::layer_size_t pool_size = 0;

        if (!get_kernel_size_2d(pool_param, &pool_size))
            pool_size = pool_param.kernel_size();

        // tiny-cnn has trainable parameter in average-pooling layer
        tiny_cnn::float_t weight = 1.0 / tiny_cnn::sqr(pool_size);
        if (!dst->weight().empty()) std::fill(dst->weight().begin(), dst->weight().end(), weight);
        if (!dst->bias().empty()) std::fill(dst->bias().begin(), dst->bias().end(), (tiny_cnn::float_t)0.0);
    }
}

inline std::shared_ptr<tiny_cnn::layer_base> create_convlayer(const caffe::LayerParameter& layer, const shape_t& bottom_shape, shape_t *top_shape) {
    using conv_layer = tiny_cnn::convolutional_layer<tiny_cnn::activation::identity>;

    if (!layer.has_convolution_param())
        throw std::runtime_error("convolution param missing");

    // layer parameters
    tiny_cnn::layer_size_t in_width = 0, in_height = 0, window_size = 0, in_channels = 0, out_channels = 0;
    tiny_cnn::layer_size_t w_stride = 1, h_stride = 1;
    bool has_bias = true;
    tiny_cnn::padding pad_type = tiny_cnn::padding::valid;

    auto conv_param = layer.convolution_param();

    // shape  
    out_channels = conv_param.num_output();
    in_channels = bottom_shape.depth_;
    in_width = bottom_shape.width_;
    in_height = bottom_shape.height_;
    has_bias = conv_param.bias_term();
    window_size = 0;

    if (!get_kernel_size_2d(conv_param, &window_size)) {
        if (conv_param.kernel_size_size() > 1)
            throw std::runtime_error("unsupported kernel shape");
        window_size = conv_param.kernel_size(0);
    }

    // padding
    if (conv_param.pad_size() == 1 || (conv_param.has_pad_w() && conv_param.has_pad_h())) {
        uint32_t pad_w = conv_param.pad_size() == 1 ? conv_param.pad(0) : conv_param.pad_w();
        uint32_t pad_h = conv_param.pad_size() == 1 ? conv_param.pad(0) : conv_param.pad_h();

        if (pad_w != pad_h)
            throw std::runtime_error("conv:not supported padding size");

        // 0 ... valid, (window_size-1)/2 ... same
        if (pad_w == (window_size - 1) / 2) {
            pad_type = tiny_cnn::padding::same;
        }
        else if (pad_w == 0) {
            pad_type = tiny_cnn::padding::valid;
        }
        else {
            throw std::runtime_error("conv:not supported padding size");
        }
    }

    // stride
    if (conv_param.stride_size() == 1 || conv_param.has_stride_h())
        h_stride = conv_param.stride_size() == 1 ? conv_param.stride(0) : conv_param.stride_h();

    if (conv_param.stride_size() == 1 || conv_param.has_stride_w())
        w_stride = conv_param.stride_size() == 1 ? conv_param.stride(0) : conv_param.stride_w();

    auto conv = std::make_shared<conv_layer>(in_width, in_height, window_size, in_channels, out_channels, pad_type, has_bias, w_stride, h_stride);

    // filler
    if (conv_param.has_weight_filler())
        conv->weight_init(create_filler(conv_param.weight_filler().type()));

    if (conv_param.has_bias_filler())
        conv->bias_init(create_filler(conv_param.bias_filler().type()));

    // set weight (optional)
    if (layer.blobs_size() > 0) { // blobs(0)...weight, blobs(1)...bias

        // fill weight
        auto weights = layer.blobs(0);

        int dim = weights.data_size();
        int curr = 0;

        for (size_t o = 0; o < out_channels; o++)
            for (size_t i = 0; i < in_channels; i++)
                for (size_t y = 0; y < window_size; y++)
                    for (size_t x = 0; x < window_size; x++)
                        conv->weight_at(i, o, x, y) = weights.data(curr++);

        // fill bias
        if (has_bias) {
            auto biases = layer.blobs(1);
            for (size_t o = 0; o < out_channels; o++)
                conv->bias()[o] = biases.data(o);
        }
    }
    *top_shape = conv->out_shape();
    return conv;
}

inline bool layer_skipped(const std::string& type) {
    if (type == "Data" || type == "EuclideanLoss") return true;
    return false;
}

inline bool layer_is_activation(const std::string& type) {
    static const char* activations[] =
    {
        "SoftmaxWithLoss", "SigmoidCrossEntropyLoss",
        "ReLU", "Sigmoid", "TanH", "Softmax"
    };
    for (int i = 0; i < sizeof(activations) / sizeof(activations[0]); i++) {
        if (activations[i] == type) return true;
    }
    return false;
}

inline bool layer_supported(const std::string& type) {
    static const char* supported[] =
    {
        "InnerProduct", "Convolution", "Pooling",
        "SoftmaxWithLoss", "SigmoidCrossEntropyLoss",
        "ReLU", "Sigmoid", "TanH", "Softmax"
    };

    for (size_t i = 0; i < sizeof(supported) / sizeof(supported[0]); i++) {
        if (supported[i] == type) return true;
    }
    return false;
}

inline bool layer_match(const std::string& caffetype, const std::string& tiny_cnn_type) {
    const char* conversions[][2] =
    {
        { "InnerProduct", "fully-connected" },
        { "Convolution", "conv" },
        { "Pooling", "ave-pool" },
        { "Pooling", "max-pool" }
    };

    for (size_t i = 0; i < sizeof(conversions) / sizeof(conversions[0]); i++) {
        if (conversions[i][0] == caffetype && conversions[i][1] == tiny_cnn_type) return true;
    }
    return false;
}

inline std::shared_ptr<tiny_cnn::layer_base> create(const caffe::LayerParameter& layer, const shape_t &in_shape, shape_t *out_shape) {
    typedef std::function<std::shared_ptr<tiny_cnn::layer_base>(const caffe::LayerParameter&, const shape_t&, shape_t*)> factoryimpl;

    std::unordered_map<std::string, factoryimpl> factory_registry;

    factory_registry["Convolution"] = ::detail::create_convlayer;
    factory_registry["InnerProduct"] = ::detail::create_fullyconnected;
    factory_registry["Pooling"] = ::detail::create_pooling;
    factory_registry["SoftmaxWithLoss"] = ::detail::create_softmax;
    factory_registry["SigmoidCrossEntropyLoss"] = ::detail::create_sigmoid;
    factory_registry["ReLU"] = ::detail::create_relu;
    factory_registry["Sigmoid"] = ::detail::create_sigmoid;
    factory_registry["TanH"] = ::detail::create_tanh;
    factory_registry["Softmax"] = ::detail::create_tanh;

    if (factory_registry.find(layer.type()) == factory_registry.end())
        throw std::runtime_error("layer parser not found");

    return factory_registry[layer.type()](layer, in_shape, out_shape);
}

inline void load(const caffe::LayerParameter& src, tiny_cnn::layer_base *dst) {
    typedef std::function<void(const caffe::LayerParameter&, tiny_cnn::layer_base*)> factoryimpl;
    std::unordered_map<std::string, factoryimpl> factory_registry;

    factory_registry["Convolution"] = ::detail::load_weights_conv;
    factory_registry["InnerProduct"] = ::detail::load_weights_fullyconnected;
    factory_registry["Pooling"] = ::detail::load_weights_pool;

    if (factory_registry.find(src.type()) == factory_registry.end())
        throw std::runtime_error("layer parser not found");

    return factory_registry[src.type()](src, dst);
}


struct layer_node {
    const caffe::LayerParameter *layer;
    const layer_node *next; // top-side
    const layer_node *prev; // bottom-side

    layer_node() : layer(0), next(0), prev(0) {}
    layer_node(const caffe::LayerParameter *l) : layer(l), next(0), prev(0) {}
};

// parse caffe net and interpret as single layer vector
class caffe_layer_vector {
public:
    caffe_layer_vector(const caffe::NetParameter& net) {
        nodes.reserve(net.layer_size());

        for (int i = 0; i < net.layer_size(); i++) {
            auto& l = net.layer(i);
            nodes.emplace_back(&l);
            layer_table[l.name()] = &nodes.back();
        }

        for (int i = 0; i < net.layer_size(); i++) {
            auto& l = nodes[i];

            if (l.layer->bottom_size() > 0 && blob_table[l.layer->bottom(0)]) {
                auto& bottom = blob_table[l.layer->bottom(0)];
                l.prev = bottom;
                layer_table[bottom->layer->name()]->next = &l;
            }

            if (l.layer->top_size() > 0) {
                blob_table[l.layer->top(0)] = &l;
            }
        }

        auto root = std::find_if(nodes.begin(), nodes.end(), [](const layer_node& n) { return n.prev == 0; });
        if (root == nodes.end())
            throw std::runtime_error("root layer not found");
        root_node = &*root;

        const layer_node *current = &*root;
        while (current) {
            node_list.push_back(current->layer);
            current = current->next;
        }
    }

    size_t size() const {
        return node_list.size();
    }

    const caffe::LayerParameter& operator [] (size_t index) const {
        return *(node_list[index]);
    }

private:
    layer_node *root_node;
    std::map<std::string, layer_node*> layer_table; // layer name -> layer
    std::map<std::string, layer_node*> blob_table; // blob name -> bottom holder
    std::vector<layer_node> nodes;
    std::vector<const caffe::LayerParameter*> node_list;
};

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////

/**
* create whole network and load weights from caffe's netparameter
*
* @param layer [in] netparameter of caffemodel
* @param data_shape [in] size of input data (width x height x channels)
*/
inline std::shared_ptr<tiny_cnn::network<tiny_cnn::mse, tiny_cnn::adagrad>>
create_net_from_caffenet(const caffe::NetParameter& layer, const tiny_cnn::layer_shape_t& data_shape)
{
::detail::caffe_layer_vector src_net(layer);
shape_t shape;

if (data_shape.size() > 0) {
    shape = data_shape;
}
else {
    if (layer.input_shape_size() == 0)
        throw std::runtime_error("input_shape not found in caffemodel. must specify input shape explicitly");
    int depth = static_cast<int>(layer.input_shape(0).dim(1));
    int width = static_cast<int>(layer.input_shape(0).dim(2));
    int height = static_cast<int>(layer.input_shape(0).dim(3));
    shape = tiny_cnn::layer_shape_t(width, height, depth);
}

auto dst_net = std::make_shared<tiny_cnn::network<tiny_cnn::mse, tiny_cnn::adagrad>>(layer.name());

for (size_t i = 0; i < src_net.size(); i++) {
    auto type = src_net[i].type();

    if (::detail::layer_skipped(type)) {
        continue;
    }

    if (!::detail::layer_supported(type))
        throw std::runtime_error("error: tiny-cnn does not support this layer type:" + type);

    shape_t shape_next = shape;
    auto layer = ::detail::create(src_net[i], shape, &shape_next);

    std::cout << "convert " << type << " => " << typeid(*layer).name() << std::endl;

    dst_net->add(layer);
    shape = shape_next;
}

return dst_net;
}

/**
* create whole network and load weights from caffe's netparameter
*
* @param layer [in] netparameter of caffemodel
* @param data_shape [in] size of input data (width x height x channels)
*/
inline std::shared_ptr<tiny_cnn::network<tiny_cnn::mse, tiny_cnn::adagrad>>
create_net_from_caffenet(const std::string& caffebinarymodel, const tiny_cnn::layer_shape_t& data_shape)
{
std::ifstream ifs(caffebinarymodel.c_str(), std::ios::in | std::ios::binary);
caffe::NetParameter np;

if (!np.ParseFromIstream(&ifs))
    throw std::runtime_error("failed to parse");

return create_net_from_caffenet(np, data_shape);
}

/**
* create whole network and load weights from caffe's netparameter
*
* @param layer [in] netparameter of caffe prototxt
*/
inline std::shared_ptr<tiny_cnn::network<tiny_cnn::mse, tiny_cnn::adagrad>>
create_net_from_caffeproto(const std::string& caffeprototxt)
{
int fd = _open(caffeprototxt.c_str(), O_RDONLY);
if (fd == -1)
    throw std::runtime_error("file not fonud: " + caffeprototxt);

caffe::NetParameter np;

google::protobuf::io::FileInputStream input(fd);
input.SetCloseOnDelete(true);

if (!google::protobuf::TextFormat::Parse(&input, &np))
    throw std::runtime_error("failed to parse");

return create_net_from_caffenet(np, tiny_cnn::layer_shape_t());
}

template <typename E, typename O>
inline void load_weight_from_caffemodel(const caffe::NetParameter& layer, tiny_cnn::network<E, O> *net)
{
::detail::caffe_layer_vector src_net(layer);

int tinycnn_layer_idx = 0;

for (int caffe_layer_idx = 0; caffe_layer_idx < src_net.size(); caffe_layer_idx++) {
    auto type = src_net[caffe_layer_idx].type();

    if (::detail::layer_skipped(type) || ::detail::layer_is_activation(type)) {
        continue;
    }

    if (!::detail::layer_supported(type))
        throw std::runtime_error("error: tiny-cnn does not support this layer type:" + type);

    while (tinycnn_layer_idx < net->depth() && !::detail::layer_match(type, (*net)[tinycnn_layer_idx]->layer_type())) {
        tinycnn_layer_idx++;
    }
    if (tinycnn_layer_idx >= net->depth()) break;

    // load weight
    ::detail::load(src_net[caffe_layer_idx], (*net)[tinycnn_layer_idx]);
}
}

template <typename E, typename O>
inline void load_weight_from_caffemodel(const std::string& caffeprototxt, tiny_cnn::network<E, O> *net)
{
int fd = _open(caffeprototxt.c_str(), O_RDONLY);
if (fd == -1)
    throw std::runtime_error("file not fonud: " + caffeprototxt);

caffe::NetParameter np;

google::protobuf::io::FileInputStream input(fd);
input.SetCloseOnDelete(true);

if (!google::protobuf::TextFormat::Parse(&input, &np))
    throw std::runtime_error("failed to parse");

return load_weight_from_caffemodel(np, net);
}

} // namespace tiny_cnn