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
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include "caffe.pb.h"

#include "tiny_dnn/network.h"
#include "tiny_dnn/lossfunctions/loss_function.h"
#include "tiny_dnn/optimizers/optimizer.h"
#include "tiny_dnn/util/util.h"
#include "tiny_dnn/io/caffe/layer_factory_impl.h"

namespace tiny_dnn {

/**
* create whole network and load weights from caffe's netparameter
*
* @param layer [in] netparameter of caffemodel
* @param data_shape [in] size of input data (width x height x channels)
*/
inline std::shared_ptr<network<sequential>>
create_net_from_caffe_net(const caffe::NetParameter& layer, const shape3d& data_shape)
{
    detail::caffe_layer_vector src_net(layer);
    shape_t shape;

    if (data_shape.size() > 0) {
        shape = data_shape;
    } else {
        if (layer.input_shape_size() > 0) {
            // input_shape is deprecated in Caffe
            // blob dimensions are ordered by number N x channel K x height H x width W
            int depth  = static_cast<int>(layer.input_shape(0).dim(1));
            int height = static_cast<int>(layer.input_shape(0).dim(2));
            int width  = static_cast<int>(layer.input_shape(0).dim(3));
            shape = shape3d(width, height, depth);
        }
        else if (src_net[0].has_input_param()) {
            // blob dimensions are ordered by number N x channel K x height H x width W
            int depth  = static_cast<int>(src_net[0].input_param().shape(0).dim(1));
            int height = static_cast<int>(src_net[0].input_param().shape(0).dim(2));
            int width  = static_cast<int>(src_net[0].input_param().shape(0).dim(3));
            shape = shape3d(width, height, depth);
        }
        else {
            throw nn_error("input_shape not found in caffemodel. must specify input shape explicitly");
        }
    }

    auto dst_net = std::make_shared<network<sequential>>(layer.name());

    for (size_t i = 0; i < src_net.size(); i++) {
        auto type = src_net[i].type();

        if (detail::layer_skipped(type)) {
            continue;
        }

        if (!detail::layer_supported(type)) {
            throw nn_error("error: tiny-dnn does not support this layer type:" + type);
        }

        shape_t shape_next = shape;
        auto layer = detail::create(src_net[i], shape, &shape_next);

        nn_info("convert " + type + " => " + typeid(*layer).name());
        nn_info("shape:" + to_string(shape_next));

        *dst_net << layer;
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
inline std::shared_ptr<network<sequential>>
create_net_from_caffe_protobinary(const std::string& caffebinarymodel, const shape3d& data_shape)
{
    caffe::NetParameter np;

    detail::read_proto_from_binary(caffebinarymodel, &np);
    return create_net_from_caffe_net(np, data_shape);
}

/**
 * create whole network and load weights from caffe's netparameter
 *
 * @param layer [in] netparameter of caffe prototxt
 */
inline std::shared_ptr<network<sequential>>
create_net_from_caffe_prototxt(const std::string& caffeprototxt, const shape3d& shape =shape3d())
{
    caffe::NetParameter np;

    detail::read_proto_from_text(caffeprototxt, &np);
    return create_net_from_caffe_net(np, shape);
}

/**
 * reload network weights from caffe's netparameter
 * this must be called after the network layers are constructed
 *
 * @param layer [in] caffe's netparameter
 * @param net [out] tiny-dnn's network
 */
template <typename N>
inline void reload_weight_from_caffe_net(const caffe::NetParameter& layer, network<N> *net)
{
    detail::caffe_layer_vector src_net(layer);

    size_t tiny_layer_idx = 0;

    for (size_t caffe_layer_idx = 0; caffe_layer_idx < src_net.size(); caffe_layer_idx++) {
        auto type = src_net[caffe_layer_idx].type();

        size_t next_idx = tiny_layer_idx + 1;

        while (next_idx < net->depth() && !detail::layer_match(type, (*net)[next_idx]->layer_type())) {
            next_idx++;
        }
        if (next_idx >= net->depth()) break;

        tiny_layer_idx = next_idx;

        // load weight
        detail::load(src_net[caffe_layer_idx], (*net)[tiny_layer_idx++]);
    }
}

/**
 * reload network weights from caffe's netparameter
 * this must be called after the network layers are constructed
 *
 * @param caffebinary [in] caffe's trained model file(binary format)
 * @param net [out] tiny-dnn's network
 */
template <typename N>
inline void reload_weight_from_caffe_protobinary(const std::string& caffebinary, network<N> *net)
{
    caffe::NetParameter np;

    detail::read_proto_from_binary(caffebinary, &np);
    reload_weight_from_caffe_net(np, net);
}

} // namespace tiny_dnn
