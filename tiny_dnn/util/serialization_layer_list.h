/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/layers/layers.h"

namespace tiny_dnn {

template <typename T>
void register_layers(T* h) {
  
#define IMPLEMENT_REGISTER_LAYER(CLASS, NAME) \
h->template register_layer<CLASS>(NAME);
#define TINYDNN_LAYER(CLASS, NAME) IMPLEMENT_REGISTER_LAYER(CLASS, NAME)
#include "layers.inc"
#undef TINYDNN_LAYER

}

}  // namespace tiny_dnn
