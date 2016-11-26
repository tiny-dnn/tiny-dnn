/*
    Copyright (c) 2016, Taiga Nomi
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
#include "tiny_dnn/layers/layer.h"

namespace tiny_dnn {

    enum class slice_type {
        slice_samples,
        slice_channels
    };


/**
 * slice an input data into multiple outputs along a given slice dimension.
 **/
class slice_layer : public layer {
public:
    typedef layer Base;

    /**
     * @param in_shape    [in] size (width * height * channels) of input data
     * @param slice_type  [in] target axis of slicing
     * @param num_outputs [in] number of output layers
     *
     * example1:
     *   input:       NxKxWxH = 4x3x2x2  (N:batch-size, K:channels, W:width, H:height)
     *   slice_type:  slice_samples
     *   num_outputs: 3
     *
     *   output[0]: 1x3x2x2
     *   output[1]: 1x3x2x2
     *   output[2]: 2x3x2x2  (mod data is assigned to the last output)
     *
     * example2:
     *   input:       NxKxWxH = 4x6x2x2
     *   slice_type:  slice_channels
     *   num_outputs: 3
     *
     *   output[0]: 4x2x2x2
     *   output[1]: 4x2x2x2
     *   output[2]: 4x2x2x2
    **/
    slice_layer(const shape3d& in_shape, slice_type slice_type, serial_size_t num_outputs)
    : layer(std::vector<vector_type>(1, vector_type::data), std::vector<vector_type>(num_outputs, vector_type::data)),
      in_shape_(in_shape), slice_type_(slice_type), num_outputs_(num_outputs) {
        set_shape();
    }

    slice_layer(const layer& prev_layer, slice_type slice_type, serial_size_t num_outputs)
        : layer(std::vector<vector_type>(1, vector_type::data), std::vector<vector_type>(num_outputs, vector_type::data)),
        in_shape_(prev_layer.out_shape()[0]), slice_type_(slice_type), num_outputs_(num_outputs) {
        set_shape();
    }

    std::string layer_type() const override {
        return "slice";
    }

    std::vector<shape3d> in_shape() const override {
        return {in_shape_};
    }

    std::vector<shape3d> out_shape() const override {
        return out_shapes_;
    }

    void forward_propagation(const std::vector<tensor_t*>& in_data,
                             std::vector<tensor_t*>& out_data) override {
        switch (slice_type_) {
        case slice_type::slice_samples:
            slice_data_forward(*in_data[0], out_data);
            break;
        case slice_type::slice_channels:
            slice_channels_forward(*in_data[0], out_data);
            break;
        default:
            throw nn_not_implemented_error();
        }
    }

    void back_propagation(const std::vector<tensor_t*>& in_data,
                          const std::vector<tensor_t*>& out_data,
                          std::vector<tensor_t*>&       out_grad,
                          std::vector<tensor_t*>&       in_grad) override {
        CNN_UNREFERENCED_PARAMETER(in_data);
        CNN_UNREFERENCED_PARAMETER(out_data);

        switch (slice_type_) {
        case slice_type::slice_samples:
            slice_data_backward(out_grad, *in_grad[0]);
            break;
        case slice_type::slice_channels:
            slice_channels_backward(out_grad, *in_grad[0]);
            break;
        default:
            throw nn_not_implemented_error();
        }
    }

    template <class Archive>
    static void load_and_construct(Archive & ar, cereal::construct<slice_layer> & construct) {
        shape3d in_shape;
        slice_type slice_type;
        serial_size_t num_outputs;

        ar(cereal::make_nvp("in_size", in_shape), cereal::make_nvp("slice_type", slice_type), cereal::make_nvp("num_outputs", num_outputs));
        construct(in_shape, slice_type, num_outputs);
    }

    template <class Archive>
    void serialize(Archive & ar) {
        layer::serialize_prolog(ar);
        ar(cereal::make_nvp("in_size", in_shape_), cereal::make_nvp("slice_type", slice_type_), cereal::make_nvp("num_outputs", num_outputs_));
    }
private:
    void slice_data_forward(const tensor_t& in_data,
                            std::vector<tensor_t*>& out_data) {
        const vec_t* in  = &in_data[0];

        for (serial_size_t i = 0; i < num_outputs_; i++) {
            tensor_t& out = *out_data[i];

            std::copy(in, in + slice_size_[i], &out[0]);

            in += slice_size_[i];
        }
    }

    void slice_data_backward(std::vector<tensor_t*>& out_grad,
                             tensor_t& in_grad) {
        vec_t* in = &in_grad[0];

        for (serial_size_t i = 0; i < num_outputs_; i++) {
            tensor_t& out = *out_grad[i];

            std::copy(&out[0], &out[0] + slice_size_[i], in);

            in += slice_size_[i];
        }
    }

    void slice_channels_forward(const tensor_t& in_data,
                                std::vector<tensor_t*>& out_data) {
        serial_size_t num_samples = static_cast<serial_size_t>(in_data.size());
        serial_size_t channel_idx = 0;
        serial_size_t spatial_dim = in_shape_.area();

        for (serial_size_t i = 0; i < num_outputs_; i++) {
            for (serial_size_t s = 0; s < num_samples; s++) {
                float_t       *out = &(*out_data[i])[s][0];
                const float_t *in  = &in_data[s][0] + channel_idx*spatial_dim;

                std::copy(in, in + slice_size_[i] * spatial_dim, out);
            }
            channel_idx += slice_size_[i];
        }
    }

    void slice_channels_backward(std::vector<tensor_t*>& out_grad,
                                 tensor_t&               in_grad) {
        serial_size_t num_samples = static_cast<serial_size_t>(in_grad.size());
        serial_size_t channel_idx = 0;
        serial_size_t spatial_dim = in_shape_.area();

        for (serial_size_t i = 0; i < num_outputs_; i++) {
            for (serial_size_t s = 0; s < num_samples; s++) {
                const float_t *out = &(*out_grad[i])[s][0];
                float_t       *in = &in_grad[s][0] + channel_idx*spatial_dim;

                std::copy(out, out + slice_size_[i] * spatial_dim, in);
            }
            channel_idx += slice_size_[i];
        }
    }

    void set_sample_count(serial_size_t sample_count) override {
        if (slice_type_ == slice_type::slice_samples) {
            if (num_outputs_ == 0)
                throw nn_error("num_outputs must be positive integer");

            serial_size_t sample_per_out = sample_count / num_outputs_;

            slice_size_.resize(num_outputs_, sample_per_out);
            slice_size_.back() = sample_count - (sample_per_out*(num_outputs_-1));
        }
        Base::set_sample_count(sample_count);
    }

    void set_shape() {
        switch (slice_type_) {
        case slice_type::slice_samples:
            set_shape_data();
            break;
        case slice_type::slice_channels:
            set_shape_channels();
            break;
        default:
            throw nn_not_implemented_error();
        }
    }

    void set_shape_data() {
        out_shapes_.resize(num_outputs_, in_shape_);
    }

    void set_shape_channels() {
        serial_size_t channel_per_out = in_shape_.depth_ / num_outputs_;

        out_shapes_.clear();
        for (serial_size_t i = 0; i < num_outputs_; i++) {
            serial_size_t ch = channel_per_out;

            if (i == num_outputs_ - 1) {
                assert(in_shape_.depth_ >= i * channel_per_out);
                ch = in_shape_.depth_ - i * channel_per_out;
            }

            slice_size_.push_back(ch);
            out_shapes_.push_back(shape3d(in_shape_.width_, in_shape_.height_, ch));
        }
    }

    shape3d in_shape_;
    slice_type slice_type_;
    serial_size_t num_outputs_;
    std::vector<shape3d> out_shapes_;
    std::vector<serial_size_t> slice_size_;
};

} // namespace tiny_dnn
