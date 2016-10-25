// Copyright (c) 2016, Taiga Nomi, Edgar Riba. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#pragma once

namespace tiny_dnn {
namespace core {

struct maxpool_params {
  index3d<cnn_size_t> in_;
  index3d<cnn_size_t> out_;
  size_t pool_size_;
  size_t stride_;
};

struct max_pooling_layer_worker_specific_storage {
  /* mapping out => max_index(in) (1:1) */
  std::vector<std::vector<cnn_size_t>> out2inmax_;
};

}  // namespace core
}  // namespace tiny_dnn
