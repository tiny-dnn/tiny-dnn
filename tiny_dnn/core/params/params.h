/*
    Copyright (c) 2016, Taiga Nomi, Edgar Riba
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

/* Base class to model operation parameters */
class Params {
 public:
  Params() {}

  conv_params conv() const;
  fully_params fully() const;
  maxpool_params &maxpool();
  global_avepool_params &global_avepool();
};

}  // namespace core
}  // namespace tiny_dnn
