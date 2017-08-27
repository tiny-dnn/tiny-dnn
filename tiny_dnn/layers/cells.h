/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once
#include "tiny_dnn/layers/gru_cell.h"
#include "tiny_dnn/layers/lstm_cell.h"
#include "tiny_dnn/layers/rnn_cell.h"

namespace tiny_dnn {
// Wrappers for the cell constructors should be placed here.

// rnn cell:
inline std::shared_ptr<cell> rnn(
  size_t in_dim,
  size_t out_dim,
  const rnn_cell_parameters params = rnn_cell_parameters()) {
  return std::make_shared<rnn_cell>(rnn_cell(in_dim, out_dim, params));
}

// gru cell:
inline std::shared_ptr<cell> gru(
  size_t in_dim,
  size_t out_dim,
  const gru_cell_parameters params = gru_cell_parameters()) {
  return std::make_shared<gru_cell>(gru_cell(in_dim, out_dim, params));
}

// lstm cell:
inline std::shared_ptr<cell> lstm(
  size_t in_dim,
  size_t out_dim,
  const lstm_cell_parameters params = lstm_cell_parameters()) {
  return std::make_shared<lstm_cell>(lstm_cell(in_dim, out_dim, params));
}
}  // namespace tiny_dnn
