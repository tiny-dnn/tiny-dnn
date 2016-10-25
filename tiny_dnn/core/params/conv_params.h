// Copyright (c) 2016, Taiga Nomi, Edgar Riba. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

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
  connection_table(const bool* ar, cnn_size_t rows, cnn_size_t cols)
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

  bool is_empty() const { return rows_ == 0 && cols_ == 0; }

  template <typename Archive>
  void serialize(Archive& ar) {
    ar(cereal::make_nvp("rows", rows_), cereal::make_nvp("cols", cols_));

    if (is_empty()) {
      ar(cereal::make_nvp("connection", std::string("all")));
    } else {
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

  friend std::ostream& operator<<(std::ostream& o,
                                  const core::conv_params& param) {
    o << "in:        " << param.in << "\n";
    o << "out:       " << param.out << "\n";
    o << "in_padded: " << param.in_padded << "\n";
    o << "weight:    " << param.weight << "\n";
    o << "has_bias:  " << param.has_bias << "\n";
    o << "w_stride:  " << param.w_stride << "\n";
    o << "h_stride:  " << param.h_stride << "\n";
    return o;
  }
};

}  // namespace core
}  // namespace tiny_dnn
