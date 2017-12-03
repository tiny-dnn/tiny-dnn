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

struct deconv_params {
  connection_table tbl;
  index3d<size_t> in;
  index3d<size_t> out;
  index3d<size_t> out_unpadded;
  index3d<size_t> weight;
  bool has_bias;
  padding pad_type;
  size_t w_stride;
  size_t h_stride;
};

}  // namespace core
}  // namespace tiny_dnn
