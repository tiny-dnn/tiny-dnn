/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

namespace tiny_dnn {
namespace core {

struct deconv_layer_worker_specific_storage {
  const tensor_t *prev_out_;
  const tensor_t *curr_out_unpadded_;
  tensor_t curr_out_buf_;
  tensor_t curr_delta_padded;
};

class deconv_params : public Params {
 public:
  connection_table tbl;
  index3d<size_t> in;
  index3d<size_t> out;
  index3d<size_t> out_unpadded;
  index3d<size_t> weight;
  bool has_bias;
  padding pad_type;
  size_t w_stride;
  size_t h_stride;

  friend std::ostream &operator<<(std::ostream &o,
                                  const core::deconv_params &param) {
    o << "in:         " << param.in << "\n";
    o << "out:        " << param.out << "\n";
    o << "out_unpadded: " << param.out_unpadded << "\n";
    o << "weight:     " << param.weight << "\n";
    o << "has_bias:   " << param.has_bias << "\n";
    o << "w_stride:   " << param.w_stride << "\n";
    o << "h_stride:   " << param.h_stride << "\n";
    return o;
  }
};

inline deconv_params &Params::deconv() {
  return *(static_cast<deconv_params *>(this));
}

}  // namespace core
}  // namespace tiny_dnn
