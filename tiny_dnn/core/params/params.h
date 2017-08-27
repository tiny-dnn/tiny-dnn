/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

namespace tiny_dnn {
namespace core {

class conv_params;
class fully_params;
class maxpool_params;
class global_avepool_params;
class gru_cell_params;
class rnn_cell_params;
class lstm_cell_params;

/* Base class to model operation parameters */
class Params {
 public:
  Params() {}

  conv_params &conv();
  fully_params &fully();
  maxpool_params &maxpool();
  global_avepool_params &global_avepool();
  gru_cell_params &gru_cell();
  rnn_cell_params &rnn_cell();
  lstm_cell_params &lstm_cell();
};

}  // namespace core
}  // namespace tiny_dnn
