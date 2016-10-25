// Copyright (c) 2016, Taiga Nomi, Edgar Riba. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#pragma once

namespace tiny_dnn {
namespace core {

class fully_params;

/* Base class to model operation parameters */
class Params {
 public:
  Params() {}

  fully_params fully() const;
};

}  // namespace core
}  // namespace tiny_dnn
