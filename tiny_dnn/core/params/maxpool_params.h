/*
    Copyright (c) 2016, Taiga Nomi, Edgar Riba
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
#include "tiny_dnn/core/params/params.h"

namespace tiny_dnn {
namespace core {

class maxpool_params : public Params {
 public:
    index3d<serial_size_t> in;
    index3d<serial_size_t> out;
    serial_size_t          pool_size_x;
    serial_size_t          pool_size_y;
    serial_size_t          stride_x;
    serial_size_t          stride_y;
    padding             pad_type;
    
    /* mapping out => max_index(in) (1:1) */
    std::vector<std::vector<serial_size_t>> out2inmax;
    /* mapping out => in (1:N) */
    std::vector<std::vector<serial_size_t> > out2in;
    /* mapping in => out (N:1) */
    std::vector<serial_size_t> in2out;
};

struct max_pooling_layer_worker_specific_storage {
    /* mapping out => max_index(in) (1:1) */
    std::vector<std::vector<serial_size_t>> out2inmax_;
};

// TODO(nyanp): can we do better here?
inline maxpool_params& Params::maxpool() {
    return *(static_cast<maxpool_params*>(this));
}

}  // namespace core
}  // namespace tiny_dnn
