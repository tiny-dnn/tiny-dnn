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
#include "tiny_dnn/util/math_functions.h"
#include "tiny_dnn/layers/layer.h"

#include <algorithm>

namespace tiny_dnn {


/**
 * Batch Normalization
 *
 * Normalize the activations of the previous layer at each batch
 **/
class batch_normalization_layer : public layer {
public:
    typedef layer Base;

    /**
    * @param prev_layer      [in] previous layer to be connected with this layer
    * @param epsilon         [in] small positive value to avoid zero-division
    * @param momentum        [in] momentum in the computation of the exponential average of the mean/stddev of the data
    * @param phase           [in] specify the current context (train/test)
    **/
    batch_normalization_layer(const layer& prev_layer,
                              float_t epsilon = 1e-5,
                              float_t momentum = 0.999,
                              net_phase phase = net_phase::train)
        : Base({ vector_type::data }, { vector_type::data }),
        in_channels_(prev_layer.out_shape()[0].depth_),
        in_spatial_size_(prev_layer.out_shape()[0].area()),
        phase_(phase),
        momentum_(momentum),
        eps_(epsilon),
        update_immidiately_(false)
    {
        init();
    }

    /**
    * @param in_spatial_size [in] spatial size (WxH) of the input data
    * @param in_channels     [in] channels of the input data
    * @param epsilon         [in] small positive value to avoid zero-division
    * @param momentum        [in] momentum in the computation of the exponential average of the mean/stddev of the data
    * @param phase           [in] specify the current context (train/test)
    **/
    batch_normalization_layer(serial_size_t in_spatial_size,
                              serial_size_t in_channels,
                              float_t epsilon = 1e-5,
                              float_t momentum = 0.999,
                              net_phase phase = net_phase::train)
        : Base({ vector_type::data }, { vector_type::data }),
        in_channels_(in_channels),
        in_spatial_size_(in_spatial_size),
        phase_(phase),
        momentum_(momentum),
        eps_(epsilon),
        update_immidiately_(false)
    {
        init();
    }

    virtual ~batch_normalization_layer(){}

    ///< number of incoming connections for each output unit
    serial_size_t fan_in_size() const override {
        return 1;
    }

    ///< number of outgoing connections for each input unit
    serial_size_t fan_out_size() const override {
        return 1;
    }

    std::vector<index3d<serial_size_t>> in_shape() const override {
        return{ index3d<serial_size_t>(in_spatial_size_, 1, in_channels_) };
    }

    std::vector<index3d<serial_size_t>> out_shape() const override {
        return{ index3d<serial_size_t>(in_spatial_size_, 1, in_channels_) };
    }

    void back_propagation(const std::vector<tensor_t*>& in_data,
                          const std::vector<tensor_t*>& out_data,
                          std::vector<tensor_t*>&       out_grad,
                          std::vector<tensor_t*>&       in_grad) override {
        tensor_t& prev_delta     = *in_grad[0];
        tensor_t& curr_delta     = *out_grad[0];
        const tensor_t& curr_out = *out_data[0];
        serial_size_t num_samples   = static_cast<serial_size_t>(curr_out.size());

        CNN_UNREFERENCED_PARAMETER(in_data);

        tensor_t delta_dot_y = curr_out;
        vec_t mean_delta_dot_y, mean_delta, mean_Y;

        for (serial_size_t i = 0; i < num_samples; i++) {
            for (serial_size_t j = 0; j < curr_out[0].size(); j++) {
                delta_dot_y[i][j] *= curr_delta[i][j];
            }
        }
        moments(delta_dot_y, in_spatial_size_, in_channels_, &mean_delta_dot_y, nullptr);
        moments(curr_delta, in_spatial_size_, in_channels_, &mean_delta, nullptr);
 
        // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
        //
        // dE(Y)/dX =
        //   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
        //     ./ sqrt(var(X) + eps)
        //
        for_i(num_samples, [&](int i) {
            for (serial_size_t j = 0; j < in_channels_; j++) {
                for (serial_size_t k = 0; k < in_spatial_size_; k++) {
                    serial_size_t index = j*in_spatial_size_ + k;

                    prev_delta[i][index]
                        = curr_delta[i][index] - mean_delta[j] - mean_delta_dot_y[j] * curr_out[i][index];

                    // stddev_ is calculated in the forward pass 
                    prev_delta[i][index] /= stddev_[j];
                }            
            }
        });
    }

    void forward_propagation(const std::vector<tensor_t*>& in_data,
        std::vector<tensor_t*>& out_data) override {
        vec_t* mean = nullptr;
        vec_t* variance = nullptr;
        tensor_t& in = *in_data[0];
        tensor_t& out = *out_data[0];

        if (phase_ == net_phase::train) {
            // calculate mean/variance from this batch in train phase
            mean = &mean_current_;
            variance = &variance_current_;
            moments(*in_data[0], in_spatial_size_, in_channels_, mean, variance);
        }
        else {
            // use stored mean/variance in test phase
            mean = &mean_;
            variance = &variance_;
        }

        // y = (x - mean) ./ sqrt(variance + eps)
        calc_stddev(*variance);

        for_i(parallelize_, in_data[0]->size(), [&](int i) {
            const float_t* inptr  = &in[i][0];
            float_t*       outptr = &out[i][0];

            for (serial_size_t j = 0; j < in_channels_; j++) {
                float_t m = (*mean)[j];

                for (serial_size_t k = 0; k < in_spatial_size_; k++) {
                    *outptr++ = (*inptr++ - m) / stddev_[j];
                }
            }
        });

        if (phase_ == net_phase::train && update_immidiately_) {
            mean_ = mean_current_;
            variance_ = variance_current_;
        }
    }

    void set_context(net_phase ctx) override
    {
        phase_ = ctx;
    }

    std::string layer_type() const override { return "batch-norm"; }

    virtual void post_update() override {
        for (serial_size_t i = 0; i < mean_.size(); i++) {
            mean_[i] = momentum_ * mean_[i] + (1 - momentum_) * mean_current_[i];
            variance_[i] = momentum_ * variance_[i] + (1 - momentum_) * variance_current_[i];
        }
    }

    virtual void save(std::ostream& os) const override {
        Base::save(os);
        for (auto m : mean_) os << m << " ";
        for (auto v : variance_) os << v << " ";
    }

    virtual void load(std::istream& is) override {
        Base::load(is);
        for (auto& m : mean_) is >> m;
        for (auto& v : variance_) is >> v;
    }

    virtual void load(const std::vector<float_t>& src, int& idx) override {
        Base::load(src, idx);
        for (auto& m : mean_) m = src[idx++];
        for (auto& v : variance_) v = src[idx++];
    }

    void update_immidiately(bool update) {
        update_immidiately_ = update;
    }

    void set_stddev(const vec_t& stddev) {
        stddev_ = stddev;
    }

    void set_mean(const vec_t& mean) {
        mean_ = mean;
    }

    void set_variance(const vec_t& variance) {
        variance_ = variance;
        calc_stddev(variance);
    }

    template <class Archive>
    static void load_and_construct(Archive & ar, cereal::construct<batch_normalization_layer> & construct) {
        shape3d in;
        serial_size_t in_spatial_size, in_channels;
        float_t eps, momentum;
        net_phase phase;
        vec_t mean, variance;
        
        ar(cereal::make_nvp("in_spatial_size", in_spatial_size),
            cereal::make_nvp("in_channels", in_channels),
            cereal::make_nvp("epsilon", eps),
            cereal::make_nvp("momentum", momentum),
            cereal::make_nvp("phase", phase),
            cereal::make_nvp("mean", mean),
            cereal::make_nvp("variance", variance));
        construct(in_spatial_size, in_channels, eps, momentum, phase);
        construct->set_mean(mean);
        construct->set_variance(variance);
    }

    template <class Archive>
    void serialize(Archive & ar) {
        layer::serialize_prolog(ar);
        ar(cereal::make_nvp("in_spatial_size", in_spatial_size_),
           cereal::make_nvp("in_channels", in_channels_),
           cereal::make_nvp("epsilon", eps_),
           cereal::make_nvp("momentum", momentum_),
           cereal::make_nvp("phase", phase_),
           cereal::make_nvp("mean", mean_),
           cereal::make_nvp("variance", variance_));
    }

    float_t epsilon() const {
        return eps_;
    }

    float_t momentum() const {
        return momentum_;
    }

private:
    void calc_stddev(const vec_t& variance) {
        for (size_t i = 0; i < in_channels_; i++) {
            stddev_[i] = sqrt(variance[i] + eps_);
        }
    }

    void init() {
        mean_current_.resize(in_channels_);
        mean_.resize(in_channels_);
        variance_current_.resize(in_channels_);
        variance_.resize(in_channels_);
        tmp_mean_.resize(in_channels_);
        stddev_.resize(in_channels_);
    }

    serial_size_t in_channels_;
    serial_size_t in_spatial_size_;

    net_phase phase_;
    float_t momentum_;
    float_t eps_;

    // mean/variance for this mini-batch
    vec_t mean_current_;
    vec_t variance_current_;

    vec_t tmp_mean_;

    // moving average of mean/variance
    vec_t mean_;
    vec_t variance_;
    vec_t stddev_;

    // for test
    bool update_immidiately_;
};

} // namespace tiny_dnn
