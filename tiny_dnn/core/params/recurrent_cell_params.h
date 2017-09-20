/*
    Copyright (c) 2016, Taiga Nomi, Edgar Riba
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <memory>

#include "tiny_dnn/activations/activation_layer.h"
#include "tiny_dnn/core/params/params.h"

namespace tiny_dnn {
namespace core {

class recurrent_cell_params : public Params {
 public:
  size_t in_size_;
  size_t out_size_;
  std::shared_ptr<activation_layer> activation_{};
  bool has_bias_;
};

inline recurrent_cell_params &Params::recurrent_cell() {
  return *(static_cast<recurrent_cell_params *>(this));
}

}  // namespace core
}  // namespace tiny_dnn
