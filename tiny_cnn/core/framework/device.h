/*
    Copyright (c) 2016, Taiga Nomi, Edgar Riba
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include "tiny_cnn/core/framework/device.fwd.h"
#include "tiny_cnn/core/framework/device_context.h"
#include "tiny_cnn/core/framework/kernel_launcher.h"

namespace tiny_cnn {

// Default base device constructor
device::device(const device_t type, const int id)
    : type_(type), id_(id) {}

// Regsiter an op to the current device
void device::register_op(const layer& op) {
    ops_.push_back(const_cast<layer*>(&op));
}

/* Public interface for a CPU device
 *
 * @param id The device identification number
 *
 * */
class serial_device : public device {
 public:
    explicit serial_device(const int device_id)
        : device(device_t::CPU, device_id) {}
};

/* Public interface for a GPU device
 *
 * @param type The device type
 * @param platform_id The platform identification number
 * @param device_id id The device identification number
 *
 * */
class ocl_device : public device {
 public:
    explicit ocl_device(const device_t type,
                        const int platform_id,
                        const int device_id)
        : device(type, device_id)
        , device_context_(platform_id, device_id) {}

 private:
    /* The device context */
    DeviceContext device_context_;

    /* A map holding kernels associated to this device.
     * The key value to retrieve the kernel is a string
     * specifying the operation type.
     */
    std::unordered_map<std::string, std::shared_ptr<KernelLauncher>> launchers_;
};

device* device::create(device_t device_type,
                       core::backend_t backend,
                       const int device_id) {
    switch (device_type) {
        case device_t::CPU:
            if (backend == core::backend_t::tiny_cnn) {
                return new serial_device(device_id);
            }
            else if (backend == core::backend_t::nnpack) {
                throw nn_error("Device with NNPACK not implemented yet");
            }
            else if (backend == core::backend_t::avx) {
                throw nn_error("Device with AVX not implemented yet");
            }
            else if (backend == core::backend_t::opencl) {
                // TODO(egdar): wehere we define this?
                const int platform_id = 0;
                return new ocl_device(device_type, platform_id, device_id);
            }
            else {
                throw nn_error("Not supported backend with CPU.");
            }
            break;
        case device_t::GPU:
            // TODO(edgar): check what to do here
            if (backend == core::backend_t::libdnn ||
                backend == core::backend_t::opencl) {
                // TODO(egdar): wehere we define this?
                const int platform_id = 0;
                return new ocl_device(device_type, platform_id, device_id);
            }
            break;
        default:
            throw nn_error("Not supported device type. Options: CPU and GPU");
            break;
    }
}

}  // namespace tiny_cnn
