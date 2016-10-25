// Copyright (c) 2016, Taiga Nomi, Edgar Riba. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#pragma once

#include <vector>

#include "tiny_dnn/core/backend.h"

namespace tiny_dnn {
namespace core {

class device {
 public:
  explicit device(const int id) : id_(id) {}

  int get_id() const { return id_; }

 private:
  int id_;
  std::vector<std::shared_ptr<backend>> backends_;
};

}  // namespace core
}  // namespace tiny_dnn
