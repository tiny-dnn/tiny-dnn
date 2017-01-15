/*
    Copyright (c) 2016, Taiga Nomi, Edgar Riba
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "params.h"

namespace tiny_dnn {
namespace core {

class fully_params : public Params {
 public:
  serial_size_t in_size_;
  serial_size_t out_size_;
  bool has_bias_;
};

// TODO(nyanp): can we do better here?
inline fully_params Params::fully() const {
  return *(static_cast<const fully_params *>(this));
}

}  // namespace core
}  // namespace tiny_dnn
