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

device::device(const device_t type, const int id)
    : type_(type), id_(id) {}

// Register an ops to the current device
void device::register_op(const std::vector<layer*>& ops) {
    for (auto op: ops) {
        if (!check_availability(op)) {
            throw nn_error("Missmatched device/backend combination.");
        }
        
        ops_.push_back(op);
        op->set_device(this);
    }
}

bool device::check_availability(layer* layer) {
    core::backend_t backend = layer->backend_type();
    switch (this->type()) {
        case device_t::CPU:
            if (backend == core::backend_t::tiny_cnn) return true;
            if (backend == core::backend_t::nnpack)   return true;
            if (backend == core::backend_t::avx)      return true;
            if (backend == core::backend_t::opencl)   return true;
            break;
        case device_t::GPU:
            if (backend == core::backend_t::libdnn)   return true;
            if (backend == core::backend_t::opencl)   return true;
            break;
        default:
            throw nn_error("Not supported device type. Options: CPU and GPU");
            break;    
    }
    return false;
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

}  // namespace tiny_cnn
