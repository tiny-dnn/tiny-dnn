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
#include "tiny_cnn/util/util.h"
#include "tiny_cnn/layers/layer.h"

namespace tiny_cnn {

/**
 * concat N layers along depth
 **/
class concat_layer : public layer {
public:
    concat_layer(const std::vector<shape3d>& in_shapes)
    : layer(std::vector<vector_type>(in_shapes.size(), vector_type::data), {vector_type::data}),
      in_shapes_(in_shapes) {
        set_outshape();
    }

    concat_layer(cnn_size_t num_args, cnn_size_t ndim)
        : layer(std::vector<vector_type>(num_args, vector_type::data), { vector_type::data }),
        in_shapes_(std::vector<shape3d>(num_args, shape3d(ndim,1,1))) {
        set_outshape();
    }

    void set_outshape() {
        out_shape_ = in_shapes_.front();
        for (size_t i = 1; i < in_shapes_.size(); i++) {
            if (in_shapes_[i].area() != out_shape_.area())
                throw nn_error("each input shapes to concat must have same WxH size");
            out_shape_.depth_ += in_shapes_[i].depth_;
        }
    }

    std::string layer_type() const override {
        return "concat";
    }

    std::vector<shape3d> in_shape() const override {
        return in_shapes_;
    }

    std::vector<shape3d> out_shape() const override {
        return {out_shape_};
    }

    void forward_propagation(const std::vector<tensor_t*>& in_data,
                             std::vector<tensor_t*>& out_data) override {
        const tensor_t& in1 = *in_data[0];
        tensor_t& out = *out_data[0];

        auto outiter = out.begin();

        for (cnn_size_t i = 0; i < in_shapes_.size(); i++)
            outiter = std::copy(in1.begin(), in1.end(), outiter);
    }

    void back_propagation(const std::vector<tensor_t*>& in_data,
                          const std::vector<tensor_t*>& out_data,
                          std::vector<tensor_t*>&       out_grad,
                          std::vector<tensor_t*>&       in_grad) override {
        CNN_UNREFERENCED_PARAMETER(in_data);
        CNN_UNREFERENCED_PARAMETER(out_data);

        tensor_t& curr_delta = *out_grad[0];
        auto src = curr_delta.begin();

        for (cnn_size_t i = 0; i < in_shapes_.size(); i++) {
            tensor_t& prev_delta = *in_grad[i];
            std::copy(src, src + prev_delta.size(), prev_delta.begin());
            src += prev_delta.size();
        }
    }

private:
    std::vector<shape3d> in_shapes_;
    shape3d out_shape_;
};

} // namespace tiny_cnn
