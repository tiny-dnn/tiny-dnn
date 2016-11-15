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

#include <string>
#include <vector>
#include <algorithm>

#include "tiny_dnn/util/util.h"
#include "tiny_dnn/util/image.h"
#include "tiny_dnn/layers/partial_connected_layer.h"
#include "tiny_dnn/activations/activation_function.h"

namespace tiny_dnn {

// forward_propagation
template <typename Activation>
void tiny_average_pooling_kernel(bool parallelize,
                                 const std::vector<tensor_t*>& in_data,
                                 std::vector<tensor_t*>&       out_data,
                                 const shape3d&                out_dim,
                                 float_t                       scale_factor,
                                 std::vector<typename partial_connected_layer<Activation>::wi_connections>& out2wi,
                                 Activation&                   h) {
 
    for_i(in_data[0]->size(), [&](size_t sample) {
        const vec_t& in = (*in_data[0])[sample];
        const vec_t& W = (*in_data[1])[0];
        const vec_t& b = (*in_data[2])[0];
        vec_t&       out = (*out_data[0])[sample];
        vec_t&       a = (*out_data[1])[sample];

        auto oarea = out_dim.area();
        size_t idx = 0;
        for (serial_size_t d = 0; d < out_dim.depth_; ++d) {
            float_t weight = W[d] * scale_factor;
            float_t bias = b[d];
            for (serial_size_t i = 0; i < oarea; ++i, ++idx) {
                const auto& connections = out2wi[idx];
                float_t value = float_t(0);
                for (auto connection : connections)// 13.1%
                    value += in[connection.second]; // 3.2%
                value *= weight;
                value += bias;
                a[idx] = value;
            }
        }

        assert(out.size() == out2wi.size());
        for (serial_size_t i = 0; i < static_cast<serial_size_t>(out2wi.size()); i++) {
            out[i] = h.f(a, i);
        }
    });
}

// back_propagation
template<typename Activation>
void tiny_average_pooling_back_kernel(const std::vector<tensor_t*>&   in_data,
                                      const std::vector<tensor_t*>&   out_data,
                                      std::vector<tensor_t*>&         out_grad,
                                      std::vector<tensor_t*>&         in_grad,
                                      const shape3d&                  in_dim,
                                      float_t                         scale_factor,
                                      std::vector<typename partial_connected_layer<Activation>::io_connections>& weight2io,
                                      std::vector<typename partial_connected_layer<Activation>::wo_connections>& in2wo,
                                      std::vector<std::vector<serial_size_t>>& bias2out) {

    for_i(in_data[0]->size(), [&](size_t sample) {
        const vec_t& prev_out   = (*in_data[0])[sample];
        const vec_t& W          = (*in_data[1])[0];
        vec_t&       dW         = (*in_grad[1])[sample];
        vec_t&       db         = (*in_grad[2])[sample];
        vec_t&       prev_delta = (*in_grad[0])[sample];
        vec_t&       curr_delta = (*out_grad[0])[sample];

        auto inarea = in_dim.area();
        size_t idx = 0;
        for (size_t i = 0; i < in_dim.depth_; ++i) {
            float_t weight = W[i] * scale_factor;
            for (size_t j = 0; j < inarea; ++j, ++idx) {
                prev_delta[idx] = weight * curr_delta[in2wo[idx][0].second];
            }
        }

        for (size_t i = 0; i < weight2io.size(); ++i) {
            const auto& connections = weight2io[i];
            float_t diff = float_t(0);

            for (auto connection : connections)
                diff += prev_out[connection.first] * curr_delta[connection.second];

            dW[i] += diff * scale_factor;
        }

        for (size_t i = 0; i < bias2out.size(); i++) {
            const std::vector<serial_size_t>& outs = bias2out[i];
            float_t diff = float_t(0);

            for (auto o : outs)
                diff += curr_delta[o];

            db[i] += diff;
        }
    });
}


/**
 * average pooling with trainable weights
 **/
template<typename Activation = activation::identity>
class average_pooling_layer : public partial_connected_layer<Activation> {
 public:
    typedef partial_connected_layer<Activation> Base;
    CNN_USE_LAYER_MEMBERS;

    /**
     * @param in_width     [in] width of input image
     * @param in_height    [in] height of input image
     * @param in_channels  [in] the number of input image channels(depth)
     * @param pool_size    [in] factor by which to downscale
     **/
    average_pooling_layer(serial_size_t in_width,
                          serial_size_t in_height,
                          serial_size_t in_channels,
                          serial_size_t pool_size)
            : average_pooling_layer(in_width, in_height, in_channels, pool_size, pool_size)
    {}

    average_pooling_layer(const shape3d& in_shape,
                          serial_size_t pool_size,
                          serial_size_t stride)
        : average_pooling_layer(in_shape.width_, in_shape.width_, in_shape.depth_, pool_size, stride)
    {}

    /**
     * @param in_width     [in] width of input image
     * @param in_height    [in] height of input image
     * @param in_channels  [in] the number of input image channels(depth)
     * @param pool_size    [in] factor by which to downscale
     * @param stride       [in] interval at which to apply the filters to the input
    **/
    average_pooling_layer(serial_size_t in_width,
                          serial_size_t in_height,
                          serial_size_t in_channels,
                          serial_size_t pool_size,
                          serial_size_t stride)
        : average_pooling_layer(in_width, in_height, in_channels, pool_size, pool_size, stride, stride, padding::valid)
    {}

    /**
    * @param in_width     [in] width of input image
    * @param in_height    [in] height of input image
    * @param in_channels  [in] the number of input image channels(depth)
    * @param pool_size_x  [in] factor by which to downscale
    * @param pool_size_y  [in] factor by which to downscale
    * @param stride_x     [in] interval at which to apply the filters to the input
    * @param stride_y     [in] interval at which to apply the filters to the input
    * @param pad_type     [in] padding mode(same/valid)
    **/
    average_pooling_layer(serial_size_t     in_width,
                          serial_size_t     in_height,
                          serial_size_t     in_channels,
                          serial_size_t     pool_size_x,
                          serial_size_t     pool_size_y,
                          serial_size_t     stride_x,
                          serial_size_t     stride_y,
                          padding        pad_type = padding::valid)
        : Base(in_width * in_height * in_channels,
            conv_out_length(in_width, pool_size_x, stride_x, pad_type) *
            conv_out_length(in_height, pool_size_y, stride_y, pad_type) * in_channels,
            in_channels, in_channels, float_t(1) / (pool_size_x * pool_size_y)),
        stride_x_(stride_x),
        stride_y_(stride_y),
        pool_size_x_(pool_size_x),
        pool_size_y_(pool_size_y),
        pad_type_(pad_type),
        in_(in_width, in_height, in_channels),
        out_(conv_out_length(in_width, pool_size_x, stride_x, pad_type),
             conv_out_length(in_height, pool_size_y, stride_y, pad_type), in_channels),
        w_(pool_size_x, pool_size_y, in_channels) {
        if ((in_width % pool_size_x) || (in_height % pool_size_y)) {
            pooling_size_mismatch(in_width, in_height, pool_size_x, pool_size_y);
        }

        init_connection(pool_size_x, pool_size_y);
    }
    std::vector<index3d<serial_size_t>> in_shape() const override {
        return { in_, w_, index3d<serial_size_t>(1, 1, out_.depth_) };
    }

    std::vector<index3d<serial_size_t>> out_shape() const override {
        return { out_, out_ };
    }

    std::string layer_type() const override { return "ave-pool"; }

    void forward_propagation(const std::vector<tensor_t*>& in_data,
                             std::vector<tensor_t*>& out_data) override {

        tiny_average_pooling_kernel<Activation>(
            parallelize_,
            in_data,
            out_data,
            out_,
            Base::scale_factor_,
            Base::out2wi_,
            Base::h_);
        
    }

    void back_propagation(const std::vector<tensor_t*>& in_data,
                          const std::vector<tensor_t*>& out_data,
                          std::vector<tensor_t*>&       out_grad,
                          std::vector<tensor_t*>&       in_grad) override {

        tensor_t& curr_delta = *out_grad[0];
        this->backward_activation(*out_grad[0], *out_data[0], curr_delta);

        tiny_average_pooling_back_kernel<Activation>(
            in_data,
            out_data,
            out_grad,
            in_grad,
            in_,
            Base::scale_factor_,
            Base::weight2io_,
            Base::in2wo_,
            Base::bias2out_);
    }

    template <class Archive>
    static void load_and_construct(Archive & ar, cereal::construct<average_pooling_layer> & construct) {
        shape3d in;
        serial_size_t stride_x, stride_y, pool_size_x, pool_size_y;
        padding pad_type;

        ar(cereal::make_nvp("in_size", in),
           cereal::make_nvp("pool_size_x", pool_size_x),
           cereal::make_nvp("pool_size_y", pool_size_y),
           cereal::make_nvp("stride_x", stride_x),
           cereal::make_nvp("stride_y", stride_y),
            cereal::make_nvp("pad_type", pad_type)
           );
        construct(in.width_, in.height_, in.depth_, pool_size_x, pool_size_y, stride_x, stride_y, pad_type);
    }

    template <class Archive>
    void serialize(Archive & ar) {
        layer::serialize_prolog(ar);
        ar(cereal::make_nvp("in_size", in_),
           cereal::make_nvp("pool_size_x", pool_size_x_),
           cereal::make_nvp("pool_size_y", pool_size_y_),
           cereal::make_nvp("stride_x", stride_x_),
           cereal::make_nvp("stride_y", stride_y_),
           cereal::make_nvp("pad_type", pad_type_)
        );
    }

    std::pair<serial_size_t, serial_size_t> pool_size() const { return std::make_pair(pool_size_x_, pool_size_y_); }

 private:
    serial_size_t stride_x_;
    serial_size_t stride_y_;
    serial_size_t pool_size_x_;
    serial_size_t pool_size_y_;
    padding pad_type_;
    shape3d in_;
    shape3d out_;
    shape3d w_;

    static serial_size_t pool_out_dim(serial_size_t in_size,
                                   serial_size_t pooling_size,
                                   serial_size_t stride) {
        return static_cast<int>(std::ceil((
            static_cast<float_t>(in_size) - pooling_size) / stride) + 1);
    }

    void init_connection(serial_size_t pooling_size_x, serial_size_t pooling_size_y) {
        for (serial_size_t c = 0; c < in_.depth_; ++c) {
            for (serial_size_t y = 0; y < in_.height_ - pooling_size_y + 1; y += stride_y_) {
                for (serial_size_t x = 0; x < in_.width_ - pooling_size_x + 1; x += stride_x_) {
                    connect_kernel(pooling_size_x, pooling_size_y, x, y, c);
                }
            }
        }

        for (serial_size_t c = 0; c < in_.depth_; ++c) {
            for (serial_size_t y = 0; y < out_.height_; ++y) {
                for (serial_size_t x = 0; x < out_.width_; ++x) {
                    this->connect_bias(c, out_.get_index(x, y, c));
                }
            }
        }
    }

    void connect_kernel(serial_size_t pooling_size_x,
                        serial_size_t pooling_size_y,
                        serial_size_t x,
                        serial_size_t y,
                        serial_size_t inc) {
        serial_size_t dymax = std::min(pooling_size_y, in_.height_ - y);
        serial_size_t dxmax = std::min(pooling_size_x, in_.width_ - x);
        serial_size_t dstx = x / stride_x_;
        serial_size_t dsty = y / stride_y_;
        serial_size_t outidx = out_.get_index(dstx, dsty, inc);
        for (serial_size_t dy = 0; dy < dymax; ++dy) {
            for (serial_size_t dx = 0; dx < dxmax; ++dx) {
                this->connect_weight(
                    in_.get_index(x + dx, y + dy, inc),
                    outidx,
                    inc);
            }
        }
    }
};

}  // namespace tiny_dnn

