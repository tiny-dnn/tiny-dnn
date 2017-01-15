/*
    Copyright (c) 2016, Taiga Nomi, Edgar Riba
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once
#include "tiny_dnn/core/params/params.h"

namespace tiny_dnn {
namespace core {

class maxpool_params : public Params {
 public:
  index3d<serial_size_t> in;
  index3d<serial_size_t> out;
  serial_size_t pool_size_x;
  serial_size_t pool_size_y;
  serial_size_t stride_x;
  serial_size_t stride_y;
  padding pad_type;

  /* mapping out => max_index(in) (1:1) */
  std::vector<std::vector<serial_size_t>> out2inmax;
  /* mapping out => in (1:N) */
  std::vector<std::vector<serial_size_t>> out2in;
  /* mapping in => out (N:1) */
  std::vector<serial_size_t> in2out;
};

struct max_pooling_layer_worker_specific_storage {
  /* mapping out => max_index(in) (1:1) */
  std::vector<std::vector<serial_size_t>> out2inmax_;
};

// TODO(nyanp): can we do better here?
inline maxpool_params &Params::maxpool() {
  return *(static_cast<maxpool_params *>(this));
}

}  // namespace core
}  // namespace tiny_dnn
