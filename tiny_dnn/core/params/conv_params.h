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

#include "params.h"

namespace tiny_dnn {
namespace core {

enum class padding {
    valid,  ///< use valid pixels of input
    same    ///< add zero-padding around input so as to keep image size
};

struct conv_layer_worker_specific_storage {
    std::vector<const vec_t*> prev_out_padded_;
    std::vector<vec_t> prev_out_buf_;
    std::vector<vec_t> prev_delta_padded_;
};

struct connection_table {
    connection_table() : rows_(0), cols_(0) {}
    connection_table(const bool *ar, cnn_size_t rows, cnn_size_t cols)
            : connected_(rows * cols), rows_(rows), cols_(cols) {
        std::copy(ar, ar + rows * cols, connected_.begin());
    }
    connection_table(cnn_size_t ngroups, cnn_size_t rows, cnn_size_t cols)
            : connected_(rows * cols, false), rows_(rows), cols_(cols) {
        if (rows % ngroups || cols % ngroups) {
            throw nn_error("invalid group size");
        }

        cnn_size_t row_group = rows / ngroups;
        cnn_size_t col_group = cols / ngroups;

        cnn_size_t idx = 0;

        for (cnn_size_t g = 0; g < ngroups; g++) {
            for (cnn_size_t r = 0; r < row_group; r++) {
                for (cnn_size_t c = 0; c < col_group; c++) {
                    idx = (r + g * row_group) * cols_ + c + g * col_group;
                    connected_[idx] = true;
                }
            }
        }
    }

    bool is_connected(cnn_size_t x, cnn_size_t y) const {
        return is_empty() ? true : connected_[y * cols_ + x];
    }

    bool is_empty() const {
        return rows_ == 0 && cols_ == 0;
    }

    template <typename Archive>
    void serialize(Archive & ar) {
        ar(cereal::make_nvp("rows", rows_), cereal::make_nvp("cols", cols_));

        if (is_empty()) {
            ar(cereal::make_nvp("connection", std::string("all")));
        }
        else {
            ar(cereal::make_nvp("connection", connected_));
        }
    }

    std::deque<bool> connected_;
    cnn_size_t rows_;
    cnn_size_t cols_;
};

class conv_params : public Params {
 public:
    connection_table tbl;
    index3d<cnn_size_t> in;
    index3d<cnn_size_t> in_padded;
    index3d<cnn_size_t> out;
    index3d<cnn_size_t> weight;
    bool has_bias;
    padding pad_type;
    size_t w_stride;
    size_t h_stride;

    friend std::ostream& operator<<(std::ostream &o,
                                    const core::conv_params& param) {
        o << "in:        " << param.in        << "\n";
        o << "out:       " << param.out       << "\n";
        o << "in_padded: " << param.in_padded << "\n";
        o << "weight:    " << param.weight    << "\n";
        o << "has_bias:  " << param.has_bias  << "\n";
        o << "w_stride:  " << param.w_stride  << "\n";
        o << "h_stride:  " << param.h_stride  << "\n";
        return o;
    }
};

}  // namespace core
}  // namespace tiny_dnn
