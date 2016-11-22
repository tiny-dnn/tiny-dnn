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

/**
 * concat N layers along depth
 *
 * @code
 * // in: [3,1,1],[3,1,1] out: [3,1,2] (in W,H,K order)
 * concat_layer l1(2,3); 
 *
 * // in: [3,2,2],[3,2,5] out: [3,2,7] (in W,H,K order)
 * concat_layer l2({shape3d(3,2,2),shape3d(3,2,5)});
 * @endcode
 **/
class concat_layer : public layer {
public:
    /**
     * @param in_shapes [in] shapes of input tensors
     */
    concat_layer(const std::vector<shape3d>& in_shapes)
    : layer(std::vector<vector_type>(in_shapes.size(), vector_type::data), {vector_type::data}),
      in_shapes_(in_shapes) {
        set_outshape();
    }

    /**
     * @param num_args [in] number of input tensors
     * @param ndim     [in] number of elements for each input
     */
    concat_layer(serial_size_t num_args, serial_size_t ndim)
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
        serial_size_t num_samples = static_cast<serial_size_t>((*out_data[0]).size());
        
        for (serial_size_t s = 0; s < num_samples; s++) {
            float_t* outs = &(*out_data[0])[s][0];
            
            for (serial_size_t i = 0; i < in_shapes_.size(); i++) {
                const float_t* ins = &(*in_data[i])[s][0];
                serial_size_t dim = in_shapes_[i].size();
                outs = std::copy(ins, ins + dim, outs);
            }
        }
    }

    void back_propagation(const std::vector<tensor_t*>& in_data,
                          const std::vector<tensor_t*>& out_data,
                          std::vector<tensor_t*>&       out_grad,
                          std::vector<tensor_t*>&       in_grad) override {
        CNN_UNREFERENCED_PARAMETER(in_data);
        CNN_UNREFERENCED_PARAMETER(out_data);

        size_t num_samples = (*out_grad[0]).size();
        
        for (size_t s = 0; s < num_samples; s++) {
            const float_t* outs = &(*out_grad[0])[s][0];
            
            for (serial_size_t i = 0; i < in_shapes_.size(); i++) {
                serial_size_t dim = in_shapes_[i].size();
                float_t* ins = &(*in_grad[i])[s][0];
                std::copy(outs, outs + dim, ins);
                outs += dim;
            }
        }
    }

    template <class Archive>
    static void load_and_construct(Archive & ar, cereal::construct<concat_layer> & construct) {
        std::vector<shape3d> in_shapes;

        ar(cereal::make_nvp("in_size", in_shapes));
        construct(in_shapes);
    }

    template <class Archive>
    void serialize(Archive & ar) {
        layer::serialize_prolog(ar);
        ar(in_shapes_);
    }

private:
    std::vector<shape3d> in_shapes_;
    shape3d out_shape_;
};

} // namespace tiny_dnn
