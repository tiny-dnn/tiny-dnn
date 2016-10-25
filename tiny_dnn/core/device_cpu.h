// Copyright (c) 2016, Taiga Nomi, Edgar Riba. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#pragma once

#include "tiny_dnn/core/device.h"

namespace tiny_dnn {
namespace core {

class cpu_device : public device {
 public:
  explicit cpu_device(const int id) : device(id) {}
};

}  // namespace core
}  // namespace tiny_dnn
