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
#include "tiny_dnn/core/kernels/tiny_maxpool_kernel.h"

namespace tiny_dnn {
namespace core {
namespace kernels {

inline void avx_maxpool_kernel(const tensor_t& in_data,
                               tensor_t&       out_data,
                               std::vector<std::vector<cnn_size_t>>& max_idx,
                               const std::vector<std::vector<cnn_size_t>>& out2in,
                               const bool layer_parallelize) {
    tiny_maxpool_kernel(in_data, out_data, max_idx, out2in, layer_parallelize);
}

inline void avx_maxpool_back_kernel(tensor_t& prev_delta,
                                    const tensor_t&  curr_delta,
                                    std::vector<std::vector<cnn_size_t>>& max_idx,
                                    const std::vector<cnn_size_t>& in2out,
                                    const bool layer_parallelize) {
    tiny_maxpool_back_kernel(prev_delta, curr_delta, max_idx, in2out, layer_parallelize);
}

}  // namespace kernels
}  // namespace core
}  // namespace tiny_dnn
