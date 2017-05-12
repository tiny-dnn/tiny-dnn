/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/util/util.h"

namespace tiny_dnn {

inline vec_t corrupt(vec_t &&in, float_t corruption_level, float_t min_value) {
  for (size_t i                            = 0; i < in.size(); i++)
    if (bernoulli(corruption_level)) in[i] = min_value;
  return in;
}

}  // namespace tiny_dnn
