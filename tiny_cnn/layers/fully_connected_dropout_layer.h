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
#include "fully_connected_layer.h"
#include "dropout.h"

namespace tiny_cnn {

// normal 
template<typename Activation>
class fully_connected_dropout_layer : public fully_connected_layer<Activation, dropout> {
public:
    fully_connected_dropout_layer(layer_size_t in_dim, layer_size_t out_dim, dropout::mode mode = dropout::per_data)
        : fully_connected_layer<Activation, dropout>(in_dim, out_dim)
    {
        this->filter_.set_mode(mode);
    }

    void set_dropout_rate(double rate) {
        this->filter_.set_dropout_rate(rate);
    }

    /**
     * set dropout-context (training-phase or test-phase)
     **/
    void set_context(dropout::context ctx) {
        this->filter_.set_context(ctx);
    }

    std::string layer_type() const override { return "dropout"; }

private:
    void post_update() override {
        this->filter_.end_batch();
    }
};

} // namespace tiny_cnn
