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
#include "tiny_dnn/util/util.h"
#include "tiny_dnn/util/image.h"
#include "tiny_dnn/activations/activation_function.h"

namespace tiny_dnn {

/**
 * applies max-pooing operaton to the spatial data
 **/
template <typename Activation = activation::identity>
class max_unpooling_layer : public feedforward_layer<Activation> {
public:
    CNN_USE_LAYER_MEMBERS;
    typedef feedforward_layer<Activation> Base;

    /**
     * @param in_width     [in] width of input image
     * @param in_height    [in] height of input image
     * @param in_channels  [in] the number of input image channels(depth)
     * @param unpooling_size [in] factor by which to upscale
     **/
    max_unpooling_layer(serial_size_t in_width,
                        serial_size_t in_height,
                        serial_size_t in_channels,
                        serial_size_t unpooling_size)
        : max_unpooling_layer(in_width, in_height, in_channels, unpooling_size, unpooling_size)
    {}

    max_unpooling_layer(const shape3d& in_size,
                        serial_size_t unpooling_size,
                        serial_size_t stride)
        : max_unpooling_layer(in_size.width_, in_size.height_, in_size.depth_, unpooling_size, unpooling_size)
    {}

    /**
     * @param in_width     [in] width of input image
     * @param in_height    [in] height of input image
     * @param in_channels  [in] the number of input image channels(depth)
     * @param unpooling_size [in] factor by which to upscale
     * @param stride       [in] interval at which to apply the filters to the input
    **/
    max_unpooling_layer(serial_size_t in_width,
                        serial_size_t in_height,
                        serial_size_t in_channels,
                        serial_size_t unpooling_size,
                        serial_size_t stride)
        : Base({vector_type::data}),
        unpool_size_(unpooling_size),
        stride_(stride),
        in_(in_width, in_height, in_channels),
        out_(unpool_out_dim(in_width, unpooling_size, stride), unpool_out_dim(in_height, unpooling_size, stride), in_channels)
    {
        //set_worker_count(CNN_TASK_SIZE);
        init_connection();
    }

    size_t fan_in_size() const override {
        return 1;
    }

    size_t fan_out_size() const override {
        return in2out_[0].size();
    }

    void forward_propagation(serial_size_t index,
                             const std::vector<vec_t*>& in_data,
                             std::vector<vec_t*>&       out_data)  override {
        const vec_t& in  = *in_data[0];
        // vec_t&       out = *out_data[0];
        vec_t&       a   = *out_data[1];
        std::vector<serial_size_t>& max_idx = max_unpooling_layer_worker_storage_[index].in2outmax_;

        for_(parallelize_, 0, in2out_.size(), [&](const blocked_range& r) {
            for (int i = r.begin(); i < r.end(); i++) {
                const auto& in_index = out2in_[i];
                a[i] = (max_idx[in_index] == i) ? in[in_index] : float_t(0);
            }
        });

        this->forward_activation(*out_data[0], *out_data[1]);
    }

    void back_propagation(serial_size_t                 index,
                          const std::vector<vec_t*>& in_data,
                          const std::vector<vec_t*>& out_data,
                          std::vector<vec_t*>&       out_grad,
                          std::vector<vec_t*>&       in_grad) override {
        vec_t&       prev_delta = *in_grad[0];
        vec_t&       curr_delta = *out_grad[1];
        std::vector<serial_size_t>& max_idx = max_unpooling_layer_worker_storage_[index].in2outmax_;

        CNN_UNREFERENCED_PARAMETER(in_data);

        this->backward_activation(*out_grad[0], *out_data[0], curr_delta);

        for_(parallelize_, 0, in2out_.size(), [&](const blocked_range& r) {
            for (int i = r.begin(); i != r.end(); i++) {
                serial_size_t outi = out2in_[i];
                prev_delta[i] = (max_idx[outi] == i) ? curr_delta[outi] : float_t(0);
            }
        });
    }

    std::vector<index3d<serial_size_t>> in_shape() const override { return {in_}; }
    std::vector<index3d<serial_size_t>> out_shape() const override { return {out_, out_}; }
    std::string layer_type() const override { return "max-unpool"; }
    size_t unpool_size() const {return unpool_size_;}

    virtual void set_worker_count(serial_size_t worker_count) override {
        Base::set_worker_count(worker_count);
        max_unpooling_layer_worker_storage_.resize(worker_count);
        for (max_unpooling_layer_worker_specific_storage& mws : max_unpooling_layer_worker_storage_) {
            mws.in2outmax_.resize(out_.size());
        }
    }

    template <class Archive>
    static void load_and_construct(Archive & ar, cereal::construct<max_unpooling_layer> & construct) {
        shape3d in;
        serial_size_t stride, unpool_size;

        ar(cereal::make_nvp("in_size", in), cereal::make_nvp("unpool_size", unpool_size), cereal::make_nvp("stride", stride));
        construct(in, unpool_size, stride);
    }

    template <class Archive>
    void serialize(Archive & ar) {
        layer::serialize_prolog(ar);
        ar(cereal::make_nvp("in_size", in_), cereal::make_nvp("unpool_size", unpool_size_), cereal::make_nvp("stride", stride_));
    }

private:
    serial_size_t unpool_size_;
    serial_size_t stride_;
    std::vector<serial_size_t> out2in_; // mapping out => in (N:1)
    std::vector<std::vector<serial_size_t> > in2out_; // mapping in => out (1:N)

    struct max_unpooling_layer_worker_specific_storage {
        std::vector<serial_size_t> in2outmax_; // mapping max_index(out) => in (1:1)
    };

    std::vector<max_unpooling_layer_worker_specific_storage> max_unpooling_layer_worker_storage_;

    index3d<serial_size_t> in_;
    index3d<serial_size_t> out_;

    static serial_size_t unpool_out_dim(serial_size_t in_size, serial_size_t unpooling_size, serial_size_t stride) {
        return (int) (float_t)in_size * stride + unpooling_size - 1;
    }

    void connect_kernel(serial_size_t unpooling_size, serial_size_t inx, serial_size_t iny, serial_size_t  c)
    {
        serial_size_t dxmax = static_cast<serial_size_t>(std::min(unpooling_size, inx * stride_ - out_.width_));
        serial_size_t dymax = static_cast<serial_size_t>(std::min(unpooling_size, iny * stride_ - out_.height_));

        for (serial_size_t dy = 0; dy < dymax; dy++) {
            for (serial_size_t dx = 0; dx < dxmax; dx++) {
                serial_size_t out_index = out_.get_index(static_cast<serial_size_t>(inx * stride_ + dx),
                                                      static_cast<serial_size_t>(iny * stride_ + dy), c);
                serial_size_t in_index = in_.get_index(inx, iny, c);

                if (in_index >= in2out_.size())
                    throw nn_error("index overflow");
                if (out_index >= out2in_.size())
                    throw nn_error("index overflow");
                out2in_[out_index] = in_index;
                in2out_[in_index].push_back(out_index);
            }
        }
    }

    void init_connection()
    {
        in2out_.resize(in_.size());
        out2in_.resize(out_.size());

        for (max_unpooling_layer_worker_specific_storage& mws : max_unpooling_layer_worker_storage_) {
            mws.in2outmax_.resize(in_.size());
        }

        for (serial_size_t c = 0; c < in_.depth_; ++c)
            for (serial_size_t y = 0; y < in_.height_; ++y)
                for (serial_size_t x = 0; x < in_.width_; ++x)
                    connect_kernel(static_cast<serial_size_t>(unpool_size_),
                                   x, y, c);
    }

};

} // namespace tiny_dnn
