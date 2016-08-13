/*
    Copyright (c) 2015, Taiga Nomi
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
#include "tiny_cnn/util/image.h"
#include "tiny_cnn/activations/activation_function.h"

namespace tiny_cnn {

/**
 * applies max-pooing operaton to the spatial data
 **/
template <typename Activation = activation::identity>
class max_pooling_layer : public feedforward_layer<Activation> {
public:
    CNN_USE_LAYER_MEMBERS;
    typedef feedforward_layer<Activation> Base;

    /**
     * @param in_width     [in] width of input image
     * @param in_height    [in] height of input image
     * @param in_channels  [in] the number of input image channels(depth)
     * @param pooling_size [in] factor by which to downscale
     **/
    max_pooling_layer(cnn_size_t in_width,
                      cnn_size_t in_height,
                      cnn_size_t in_channels,
                      cnn_size_t pooling_size)
        : Base({vector_type::data}),
        pool_size_(pooling_size),
        stride_(pooling_size),
        in_(in_width, in_height, in_channels),
        out_(in_width / pooling_size, in_height / pooling_size, in_channels)
    {
        if ((in_width % pooling_size) || (in_height % pooling_size)) {
            pooling_size_mismatch(in_width, in_height, pooling_size);
        }

        //set_worker_count(CNN_TASK_SIZE);
        init_connection();
    }

    /**
     * @param in_width     [in] width of input image
     * @param in_height    [in] height of input image
     * @param in_channels  [in] the number of input image channels(depth)
     * @param pooling_size [in] factor by which to downscale
     * @param stride       [in] interval at which to apply the filters to the input
    **/
    max_pooling_layer(cnn_size_t in_width,
                      cnn_size_t in_height,
                      cnn_size_t in_channels,
                      cnn_size_t pooling_size,
                      cnn_size_t stride)
        : Base({vector_type::data}),
        pool_size_(pooling_size),
        stride_(stride),
        in_(in_width, in_height, in_channels),
        out_(pool_out_dim(in_width, pooling_size, stride), pool_out_dim(in_height, pooling_size, stride), in_channels)
    {
        //set_worker_count(CNN_TASK_SIZE);
        init_connection();
    }

    size_t fan_in_size() const override {
        return out2in_[0].size();
    }

    size_t fan_out_size() const override {
        return 1;
    }

    void forward_propagation(const std::vector<tensor_t*>& in_data,
                             std::vector<tensor_t*>&       out_data)  override {

        // @todo revise the parallelism strategy
        for (cnn_size_t sample = 0, sample_count = in_data.size(); sample < sample_count; ++sample) {

            const vec_t& in  = (*in_data[0])[sample];
            vec_t&       out = (*out_data[0])[sample];
            vec_t&       a   = (*out_data[1])[sample];

            std::vector<cnn_size_t>& max_idx = out2inmax_;

            for_(parallelize_, 0, out2in_.size(), [&](const blocked_range& r) {
                for (int i = r.begin(); i < r.end(); i++) {
                    const auto& in_index = out2in_[i];
                    float_t max_value = std::numeric_limits<float_t>::lowest();

                    for (auto j : in_index) {
                        if (in[j] > max_value) {
                            max_value = in[j];
                            max_idx[i] = j;
                        }
                    }
                    a[i] = max_value;
                }
            });

            for_i(parallelize_, out.size(), [&](int i) {
                out[i] = h_.f(a, i);
            });
        }
    }

    void back_propagation(const std::vector<tensor_t*>& in_data,
                          const std::vector<tensor_t*>& out_data,
                          std::vector<tensor_t*>&       out_grad,
                          std::vector<tensor_t*>&       in_grad) override {
        tensor_t& prev_delta = *in_grad[0];
        tensor_t& curr_delta = *out_grad[1];
        std::vector<cnn_size_t>& max_idx = out2inmax_;

        CNN_UNREFERENCED_PARAMETER(in_data);

        this->backward_activation(*out_grad[0], *out_data[0], curr_delta);

        // @todo consider revising the parallelism strategy
        for (cnn_size_t sample = 0, sample_count = in_grad[0]->size(); sample < sample_count; ++sample) {
            for_(parallelize_, 0, in2out_.size(), [&](const blocked_range& r) {
                for (int i = r.begin(); i != r.end(); i++) {
                    cnn_size_t outi = in2out_[i];
                    prev_delta[sample][i] = (max_idx[outi] == i) ? curr_delta[sample][outi] : float_t(0);
                }
            });
        }
    }

    std::vector<index3d<cnn_size_t>> in_shape() const override { return {in_}; }
    std::vector<index3d<cnn_size_t>> out_shape() const override { return {out_, out_}; }
    std::string layer_type() const override { return "max-pool"; }
    size_t pool_size() const {return pool_size_;}

private:
    size_t pool_size_;
    size_t stride_;
    std::vector<std::vector<cnn_size_t> > out2in_; // mapping out => in (1:N)
    std::vector<cnn_size_t> in2out_; // mapping in => out (N:1)
    std::vector<cnn_size_t> out2inmax_; // mapping out => max_index(in) (1:1)

    index3d<cnn_size_t> in_;
    index3d<cnn_size_t> out_;

    static cnn_size_t pool_out_dim(cnn_size_t in_size, cnn_size_t pooling_size, cnn_size_t stride) {
        return (int) std::ceil(((double)in_size - pooling_size) / stride) + 1;
    }

    void connect_kernel(cnn_size_t pooling_size, cnn_size_t outx, cnn_size_t outy, cnn_size_t  c)
    {
        cnn_size_t dxmax = static_cast<cnn_size_t>(std::min((size_t)pooling_size, in_.width_ - outx * stride_));
        cnn_size_t dymax = static_cast<cnn_size_t>(std::min((size_t)pooling_size, in_.height_ - outy * stride_));

        for (cnn_size_t dy = 0; dy < dymax; dy++) {
            for (cnn_size_t dx = 0; dx < dxmax; dx++) {
                cnn_size_t in_index = in_.get_index(static_cast<cnn_size_t>(outx * stride_ + dx),
                                                      static_cast<cnn_size_t>(outy * stride_ + dy), c);
                cnn_size_t out_index = out_.get_index(outx, outy, c);

                if (in_index >= in2out_.size())
                    throw nn_error("index overflow");
                if (out_index >= out2in_.size())
                    throw nn_error("index overflow");
                in2out_[in_index] = out_index;
                out2in_[out_index].push_back(in_index);
            }
        }
    }

    void init_connection()
    {
        in2out_.resize(in_.size());
        out2in_.resize(out_.size());
		out2inmax_.resize(out_.size());

        for (cnn_size_t c = 0; c < in_.depth_; ++c)
            for (cnn_size_t y = 0; y < out_.height_; ++y)
                for (cnn_size_t x = 0; x < out_.width_; ++x)
                    connect_kernel(static_cast<cnn_size_t>(pool_size_),
                                   x, y, c);
    }

};

} // namespace tiny_cnn
