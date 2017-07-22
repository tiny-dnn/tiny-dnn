/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <memory>

#if defined(USE_OPENCL) || defined(USE_CUDA)
#ifdef USE_OPENCL
#include "third_party/CLCudaAPI/clpp11.h"
#else
#include "third_party/CLCudaAPI/cupp11.h"
#endif
#endif

namespace tiny_dnn {

enum class device_t { NONE, CPU, GPU /*, FPGA */ };

inline std::ostream &operator<<(std::ostream &os, device_t type) {
  switch (type) {
    case device_t::NONE: os << "NONE"; break;
    case device_t::CPU: os << "CPU"; break;
    case device_t::GPU: os << "GPU"; break;
    default:
      throw nn_error("Not supported ostream enum: " +
                     to_string(static_cast<int>(type)));
      break;
  }
  return os;
}

/* The class models a physical device */
class Device {
 public:
  /* Custom CPU constructor
   *
   * @param type The device type. Can be only CPU.
   */
  inline explicit Device(device_t type);

  /* CPU/GPU OpenCL constructor.
   * Device context is initialized in constructor.
   *
   * @param type The device type. Can be both CPU and GPU.
   * @param platform_id The platform identification number.
   * @param device_id The device identification number.
   */
  inline explicit Device(device_t type,
                         const int platform_id,
                         const int device_id);

  // Returns the device type
  device_t type() const { return type_; }

  // Returns true if CLCudaAPI is enabled to this device
  bool hasCLCudaAPI() const { return has_clcuda_api_; }

  // Returns the platform id
  int platformId() const { return platform_id_; }

  // Returns the device id
  int deviceId() const { return device_id_; }

#if defined(USE_OPENCL) || defined(USE_CUDA)
  // Returns the CLCudaAPI Device object
  CLCudaAPI::Device device() const { return *device_; }

  // Returns the CLCudaAPI Context object
  CLCudaAPI::Context context() const { return *context_; }

  // Returns the CLCudaAPI Queue object
  CLCudaAPI::Queue queue() const { return *queue_; }
#endif

  bool operator==(const Device &d) const {
    if (d.type() == this->type() && d.hasCLCudaAPI() == this->hasCLCudaAPI() &&
        d.platformId() == this->platformId() &&
        d.deviceId() == this->deviceId()) {
      return true;
    }
    return false;
  }

  /* Registers and create an OpenCL program per Operation type.
   *
   * @param l The layer to be registered
   */
  inline void registerOp(layer &l);

 private:
  /* The device type */
  device_t type_;
  /* Boolean to check if device has OpenCL */
  bool has_clcuda_api_;
  /* The platform identification number */
  int platform_id_;
  /* The device identification number */
  int device_id_;
#if defined(USE_OPENCL) || defined(USE_CUDA)
  /* The CLCudaAPI device */
  std::shared_ptr<CLCudaAPI::Device> device_;
  /* The CLCudaAPI device context */
  std::shared_ptr<CLCudaAPI::Context> context_;
  /* The CLCudaAPI device queue */
  std::shared_ptr<CLCudaAPI::Queue> queue_;
#endif
};

}  // namespace tiny_dnn
