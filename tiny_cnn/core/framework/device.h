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

#include "tiny_cnn/layers/layer.fwd.h"

#ifdef USE_OPENCL
#include "third_party/CLCudaAPI/clpp11.h"
#endif

namespace tiny_cnn {

/* Supported devices type
 *
 * */
enum device_t { CPU, GPU /*, FPGA*/ };

/* Base class modeling a device 
 *
 * @param type The type of the device
 * @param id The identification number
 *
 * */
class device {
 public:
    explicit device(const device_t type, const int id)
        : type_(type), id_(id) {}

    // Register an ops to the current device
    void register_op(const std::vector<layer*>& ops) {
        for (auto op: ops) {
            if (!check_availability(op)) {
                nn_error("Missmatch device/backend combination");
            }
            
            ops_.push_back(op);
            // o->set_device(this);
        }
    }

    // Inits the device context
    void init() {

    }

    // Returns the device type
    device_t type() const { return type_; }

    // Returns the device id
    int id() const { return id_; }

    // Returns the ids list
    // TODO(edgar/naibaf7): What does it really mean
    //  this values?
    int id_list() const { return id_; }

    // Returns the device linked ops
    std::vector<layer*> ops() const { return ops_; }

 private:
    bool check_availability(layer* layer) {
        core::backend_t backend = layer->backend_type();
        switch (this->type()) {
            case device_t::CPU:
                if (backend == core::backend_t::tiny_cnn) return true;
                if (backend == core::backend_t::nnpack)   return true;
                if (backend == core::backend_t::avx)      return true;
                break;
            case device_t::GPU:
                if (backend == core::backend_t::libdnn)   return true;
                break;
            default:
                    nn_error("Not supported device type. Options; CPU and GPU");
                break;    
        }
        return false;
    }

    /* The type of the device */
    device_t type_;
    
    /* The id of the current device */
    int id_;

    /* A vector of pointers to registered ops.
     * The data is not owned by the current class.
     * */
    std::vector<layer*> ops_;
};


/* Public interface for a CPU device
 *
 * @param id The identification number
 *
 * */
class cpu_device : public device {
 public:
    explicit cpu_device(const int id)
        : device(device_t::CPU, id) {}
};

/* Public interface for a GPU device
 *
 * @param id The identification number
 *
 * */
/*class gpu_device : public device {
 public:
#ifndef USE_OPENCL
    explicit gpu_device(const in id)
            : device(device_t::GPU, id) {
        nn_error("Not compiled with OpenCL");
    }
#else
    // Initializes the CLCudaAPI platform and device. This initializes the OpenCL/CUDA back-end and
    // selects a specific device on the platform. The device class has methods to retrieve properties
    // such as the device name and vendor.
    //
    // Creates a new CLCudaAPI context and queue for this device. The queue can be used to schedule
    // commands such as launching a kernel or performing a device-host memory copy.

    explicit gpu_device(const int id)
            : device(device_t::GPU, id)
            , platform_(CLCudaAPI::Platform(size_t{0}))
            , device_(CLCudaAPI::Device(platform_, id_))
            , context_(CLCudaAPI::Context(device_))
            , queue_(CLCudaAPI::Queue(context_, device_)) {

        printf("\n## Initializing...\n");
        printf(" > Running on device '%s' of '%s'\n",
                device_.Name().c_str(), device_.Vendor().c_str());
    }

    // Returns C++11 device platform
    CLCudaAPI::Platform platform() const { return platform_; }

    // Returns C++11 device
    CLCudaAPI::Device device() const { return device_; }

    // Returns C++11 context
    CLCudaAPI::Context context() const { return context_; }

    // Returns C++11 queue
    CLCudaAPI::Queue queue() const { return queue_; }

 private:
    CLCudaAPI::Platform platform_;
    CLCudaAPI::Device device_;
    CLCudaAPI::Context context_;
    CLCudaAPI::Queue queue_;
#endif  // USE_OPENCL
};*/

}  // namespace tiny_cnn
