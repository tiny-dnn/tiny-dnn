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

#include "tiny_cnn/util/util.h"
#include "tiny_cnn/util/image.h"
#include "tiny_cnn/layers/partial_connected_layer.h"
#include "tiny_cnn/activations/activation_function.h"

namespace tiny_cnn {

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
     * @param pooling_size [in] factor by which to downscale
     **/
    average_pooling_layer(cnn_size_t in_width,
                          cnn_size_t in_height,
                          cnn_size_t in_channels,
                          cnn_size_t pooling_size)
            : Base(in_width * in_height * in_channels,
                   in_width * in_height * in_channels / sqr(pooling_size),
                   in_channels, in_channels, float_t(1) / sqr(pooling_size)),
              stride_(pooling_size),
              in_(in_width, in_height, in_channels),
              out_(in_width/pooling_size, in_height/pooling_size, in_channels),
              w_(pooling_size, pooling_size, in_channels) {
        if ((in_width % pooling_size) || (in_height % pooling_size)) {
            pooling_size_mismatch(in_width, in_height, pooling_size);
        }

        init_connection(pooling_size);
    }

    /**
     * @param in_width     [in] width of input image
     * @param in_height    [in] height of input image
     * @param in_channels  [in] the number of input image channels(depth)
     * @param pooling_size [in] factor by which to downscale
     * @param stride       [in] interval at which to apply the filters to the input
    **/
    average_pooling_layer(cnn_size_t in_width,
                          cnn_size_t in_height,
                          cnn_size_t in_channels,
                          cnn_size_t pooling_size,
                          cnn_size_t stride)
        : Base(in_width * in_height * in_channels,
               pool_out_dim(in_width, pooling_size, stride) *
               pool_out_dim(in_height, pooling_size, stride) * in_channels,
               in_channels, in_channels, float_t(1) / sqr(pooling_size)),
          stride_(stride),
          in_(in_width, in_height, in_channels),
          out_(pool_out_dim(in_width, pooling_size, stride),
               pool_out_dim(in_height, pooling_size, stride), in_channels),
          w_(pooling_size, pooling_size, in_channels) {
        if ((in_width % pooling_size) || (in_height % pooling_size)) {
            pooling_size_mismatch(in_width, in_height, pooling_size);
        }

        init_connection(pooling_size);
    }

    std::vector<shape3d> in_shape() const override {
        return { in_, w_, shape3d(1, 1, out_.depth_) };
    }

    std::vector<shape3d> out_shape() const override {
        return { out_, out_ };
    }

    std::string layer_type() const override { return "ave-pool"; }

    void forward_propagation(cnn_size_t index,
                             const std::vector<vec_t*>& in_data,
                             std::vector<vec_t*>& out_data) override {
        const vec_t& in  = *in_data[0];
        const vec_t& w   = *in_data[1];
        const vec_t& b   = *in_data[2];
        vec_t&       out = *out_data[0];
        vec_t&       a   = *out_data[1];

        CNN_UNREFERENCED_PARAMETER(index);

		auto oarea = out_.area();
		size_t idx = 0;
		for (size_t d=0; d<out_.depth_; ++d) {
			float weight = w[d] * scale_factor_;
			float bias = b[d];
			for (size_t i=0; i<oarea; ++i, ++idx) {
				const wi_connections& connections = out2wi_[idx];
				float_t value = float_t(0);
				for (auto connection : connections)// 13.1%
					value += in[connection.second]; // 3.2%
				value *= weight;
				value += bias;
				a[idx] = value;
			}
		}

		assert(out.size() == out2wi_.size());
		h_.f(out, a);
    }

    void back_propagation(cnn_size_t                 index,
                          const std::vector<vec_t*>& in_data,
                          const std::vector<vec_t*>& out_data,
                          std::vector<vec_t*>&       out_grad,
                          std::vector<vec_t*>&       in_grad) override {
        const vec_t& prev_out = *in_data[0];
        const vec_t& w  = *in_data[1];
        vec_t&       dW = *in_grad[1];
        vec_t&       db = *in_grad[2];
        vec_t&       prev_delta = *in_grad[0];
        vec_t&       curr_delta = *out_grad[0];

        CNN_UNREFERENCED_PARAMETER(index);

        this->backward_activation(*out_grad[0], *out_data[0], curr_delta);

		auto inarea = in_.area();
		size_t idx = 0;
		for (size_t i=0; i<in_.depth_; ++i) {
			float_t weight = w[i] * scale_factor_;
			for (size_t j=0; j<inarea; ++j, ++idx) {
				prev_delta[idx] = weight * curr_delta[in2wo_[idx].second];
			}
		}

		for (size_t i=0; i<weight2io_.size(); ++i) {
            const io_connections& connections = weight2io_[i];
            float_t diff = float_t(0);

            for (auto connection : connections) // 11.9%
                diff += prev_out[connection.first] * curr_delta[connection.second];

            dW[i] += diff * scale_factor_;
		}

        for (size_t i = 0; i < bias2out_.size(); i++) {
            const std::vector<cnn_size_t>& outs = bias2out_[i];
            float_t diff = float_t(0);

            for (auto o : outs)
                diff += curr_delta[o];    

            db[i] += diff;
        }
    }

 private:
    size_t stride_;
    shape3d in_;
    shape3d out_;
    shape3d w_;

    static cnn_size_t pool_out_dim(cnn_size_t in_size,
                                   cnn_size_t pooling_size,
                                   cnn_size_t stride) {
        return static_cast<int>(std::ceil((
            static_cast<double>(in_size) - pooling_size) / stride) + 1);
    }

    void init_connection(cnn_size_t pooling_size) {
        for (cnn_size_t c = 0; c < in_.depth_; ++c) {
            for (cnn_size_t y = 0; y < in_.height_ - pooling_size + 1; y += stride_) {
                for (cnn_size_t x = 0; x < in_.width_ - pooling_size + 1; x += stride_) {
                    connect_kernel(pooling_size, x, y, c);
                }
            }
        }

        for (cnn_size_t c = 0; c < in_.depth_; ++c) {
            for (cnn_size_t y = 0; y < out_.height_; ++y) {
                for (cnn_size_t x = 0; x < out_.width_; ++x) {
                    this->connect_bias(c, out_.get_index(x, y, c));
                }
            }
        }
    }

    void connect_kernel(cnn_size_t pooling_size,
                        cnn_size_t x,
                        cnn_size_t y,
                        cnn_size_t inc) {
        cnn_size_t dymax = std::min(pooling_size, in_.height_ - y);
        cnn_size_t dxmax = std::min(pooling_size, in_.width_ - x);
        cnn_size_t dstx = x / stride_;
        cnn_size_t dsty = y / stride_;

        for (cnn_size_t dy = 0; dy < dymax; ++dy) {
            for (cnn_size_t dx = 0; dx < dxmax; ++dx) {
                this->connect_weight(
                    in_.get_index(x + dx, y + dy, inc),
                    out_.get_index(dstx, dsty, inc),
                    inc);
            }
        }
    }
};

}  // namespace tiny_cnn
