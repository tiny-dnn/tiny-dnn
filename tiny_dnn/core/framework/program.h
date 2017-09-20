/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <string>

#include "tiny_dnn/core/framework/device.fwd.h"
#include "tiny_dnn/layers/layer.h"

#if defined(USE_OPENCL) || defined(USE_CUDA)
#ifdef USE_OPENCL
#include "third_party/CLCudaAPI/clpp11.h"
#else
#include "third_party/CLCudaAPI/cupp11.h"
#endif
#endif

namespace tiny_dnn {

/* The class models a program to be stored in the register.
 * Each instance of this class will be used as key in the register.
 */
class Program {
 public:
  explicit Program(const Device *device, const layer *op)
    : device_(device), op_(op) {}

  // Returns the device associated to the program
  const Device *device() const { return device_; }

  // Return the layer pointer
  const layer *op() const { return op_; }

  bool operator==(const Program &p) const {
    if (p.device() == this->device() &&
        p.op()->layer_type() == this->op()->layer_type()) {
      return true;
    }
    return false;
  }

 private:
  const Device *device_;
  const layer *op_;
};

/* Hash function to store Programs in the register.
 */
class ProgramHash {
 public:
  size_t operator()(const Program &p) const {
    // check there is a device and an op assigned
    // to the input program.
    if (p.device() == nullptr || p.op() == nullptr) {
      throw nn_error("No Op or Device in Program.");
    }

    // Compute individual hash values for data members and combine
    // them using XOR and bit shifting.
    return (std::hash<int>()(static_cast<int>(p.device()->type())) ^
            std::hash<bool>()(p.device()->hasCLCudaAPI()) ^
            std::hash<int>()(p.device()->platformId()) ^
            std::hash<int>()(p.device()->deviceId()) ^
            std::hash<std::string>()(p.op()->layer_type()));
  }
};

}  // namespace tiny_dnn
