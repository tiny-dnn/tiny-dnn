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

// optimized for 2x2 average pooling layer

/**
 * average pooling with trainable weights
 **/
template<typename Activation = activation::identity>
class average_pooling_layer_2x2 : public partial_connected_layer<Activation> {
 public:
    typedef partial_connected_layer<Activation> Base;
    CNN_USE_LAYER_MEMBERS;

    /**
     * @param in_width     [in] width of input image
     * @param in_height    [in] height of input image
     * @param in_channels  [in] the number of input image channels(depth)
     * @param pooling_size [in] factor by which to downscale
     **/
    average_pooling_layer_2x2(cnn_size_t in_width,
                              cnn_size_t in_height,
                              cnn_size_t in_channels,
                              cnn_size_t pooling_size)
            : Base(in_width * in_height * in_channels,
                   in_width * in_height * in_channels / sqr(pooling_size),
                   in_channels, in_channels, float_t(1) / sqr(pooling_size)),
              in_(in_width, in_height, in_channels),
              out_(in_width/pooling_size, in_height/pooling_size, in_channels),
              w_(pooling_size, pooling_size, in_channels) {
        
        assert(pooling_size == 2);
        if ((in_width % pooling_size) || (in_height % pooling_size)) {
            pooling_size_mismatch(in_width, in_height, pooling_size);
        }

    }

    /**
     * @param in_width     [in] width of input image
     * @param in_height    [in] height of input image
     * @param in_channels  [in] the number of input image channels(depth)
     * @param pooling_size [in] factor by which to downscale
     * @param stride       [in] interval at which to apply the filters to the input
    **/
    average_pooling_layer_2x2(cnn_size_t in_width,
                              cnn_size_t in_height,
                              cnn_size_t in_channels,
                              cnn_size_t pooling_size,
                              cnn_size_t stride)
        : Base(in_width * in_height * in_channels,
               pool_out_dim(in_width, pooling_size, stride) *
               pool_out_dim(in_height, pooling_size, stride) * in_channels,
               in_channels, in_channels, float_t(1) / sqr(pooling_size)),
          in_(in_width, in_height, in_channels),
          out_(pool_out_dim(in_width, pooling_size, stride),
               pool_out_dim(in_height, pooling_size, stride), in_channels),
          w_(pooling_size, pooling_size, in_channels) {
        
        assert(pooling_size == 2);
        if ((in_width % pooling_size) || (in_height % pooling_size)) {
            pooling_size_mismatch(in_width, in_height, pooling_size);
        }

    }
    
    size_t fan_in_size() const override {
        return 4;
    }
    
    size_t fan_out_size() const override {
        return 1;
    }

    std::vector<index3d<cnn_size_t>> in_shape() const override {
        return { in_, w_, index3d<cnn_size_t>(1, 1, out_.depth_) };
    }

    std::vector<index3d<cnn_size_t>> out_shape() const override {
        return { out_, out_ };
    }

    std::string layer_type() const override { return "ave-pool"; }

    void forward_propagation(cnn_size_t index,
                             const std::vector<fvec_t*>& in_data,
                             std::vector<fvec_t*>& out_data) {
        const fvec_t& in  = *in_data[0];
        const fvec_t& w   = *in_data[1];
        const fvec_t& b   = *in_data[2];
        fvec_t&       out = *out_data[0];
        fvec_t&       a   = *out_data[1];

        CNN_UNREFERENCED_PARAMETER(index);

        for_i(parallelize_, out2wi_.size(), [&](int i) {
            const wi_connections& connections = out2wi_[i];

			float value = float(0);

            for (auto connection : connections)// 13.1%
                value += w[connection.first] * in[connection.second]; // 3.2%

            value *= scale_factor_;
            value += b[out2bias_[i]];
            a[i] = value;
        });
		assert(out.size() == out2wi_.size());
		h_.f(out, a);
    }

    void back_propagation(cnn_size_t                 index,
                          const std::vector<fvec_t*>& in_data,
                          const std::vector<fvec_t*>& out_data,
                          std::vector<fvec_t*>&       out_grad,
                          std::vector<fvec_t*>&       in_grad) {
        const fvec_t& prev_out = *in_data[0];
        const fvec_t& w  = *in_data[1];
        fvec_t&       dW = *in_grad[1];
        fvec_t&       db = *in_grad[2];
        fvec_t&       prev_delta = *in_grad[0];
        fvec_t&       curr_delta = *out_grad[0];

        CNN_UNREFERENCED_PARAMETER(index);

        this->backward_activation(*out_grad[0], *out_data[0], curr_delta);

        for_(parallelize_, 0, in2wo_.size(), [&](const blocked_range& r) {
            for (int i = r.begin(); i != r.end(); i++) {
                const auto& connection = in2wo_[i];
                float delta = w[connection.first] * curr_delta[connection.second]; // 40.6%
                prev_delta[i] = delta * scale_factor_; // 2.1%
            }
        });

        for_(parallelize_, 0, weight2io_.size(), [&](const blocked_range& r) {
            for (int i = r.begin(); i < r.end(); i++) {
                const io_connections& connections = weight2io_[i];
                float diff = float(0);

                for (auto connection : connections) // 11.9%
                    diff += prev_out[connection.first] * curr_delta[connection.second];

                dW[i] += diff * scale_factor_;
            }
        });

        for (size_t i = 0; i < bias2out_.size(); i++) {
            const std::vector<cnn_size_t>& outs = bias2out_[i];
            float diff = float(0);

            for (auto o : outs)
                diff += curr_delta[o];    

            db[i] += diff;
        } 
    }

 private:
    shape3d in_;
    shape3d out_;
    shape3d w_;

    static cnn_size_t pool_out_dim(cnn_size_t in_size,
                                   cnn_size_t pooling_size,
                                   cnn_size_t stride) {
        return static_cast<int>(std::ceil((
            static_cast<double>(in_size) - pooling_size) / stride) + 1);
    }

};

}  // namespace tiny_cnn
